# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:50:15 2019

20191008 : plot_1d support xran

mcstas中的I.I换算到中子通量n/s
    100KW : 3.9E14
    500KW : 1.56E15
如果换算到单位面积，再除以面积

@author: luowei
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as scipy_cons
m_value = scipy_cons.physical_constants['neutron mass'][0] # kg
meV = scipy_cons.eV/10**3  # 

#======================================================================================
def select_interp(x,df,index_x='',index_y =''):
    # linear interpolate
    # must sort values first !!!!
    data = df.sort_values(by=index_x).copy()
    return np.interp(x,data[index_x],data[index_y])

def select_from_sections(x,df,index_x=''):
    # select single energy data
    junk = df.loc[x <= df[index_x]]
    return junk.iloc[0]

#======================================================================================
def split_first(string,sep = ':'):
    # split方法的扩展,只split第一个sep，同时都strip处理
    junk_s = string.split(sep)
    junk1 = junk_s[1]
    for i in range(len(junk_s)-2):
        junk1 = junk1+sep+junk_s[i+2]
    return [junk_s[0].strip(),junk1.strip()]

class Read_Simu():
    # 读取mcstas模拟数据中的mccode.sim文件信息
    # 由于
    def __init__(self,path):
        self.path = path
        self.filename = path + 'mccode.sim'
        self.read_index()
        self.read_simu()
    
    def read_simu(self):
        info = {}
        info['header'] = self.read_dict(0)
        info['instrument'] = self.read_dict(1)
        info['simulation'] = self.read_dict(2)
        # read data
        for i in range(self.n_ind-3):
            junk = self.read_dict(i+3)
            info[junk['component']] = junk
        self.info = info
    
    def read_index(self):
        f = open(self.filename,'r')
        self.junk_se = pd.Series(f.readlines())
        index = self.junk_se[self.junk_se == '\n'].index.values
        ind_end = np.array(list(index[1:]-1)+[len(self.junk_se)-1])
        self.n_ind = len(index) + 1

        self.ind_title = [0]+list(index+1)
        self.ind_dict = [[1,2]]
        for i in range(self.n_ind-1):
            self.ind_dict = self.ind_dict + [[index[i]+2,ind_end[i]-1]]
        f.close()
    
    def read_dict(self,ind_numb):
        # ind_numb : 0,1,2,...
        header = {}
        key = ''
        value = ''
        ind_start = self.ind_dict[ind_numb][0]
        ind_end = self.ind_dict[ind_numb][1]+1
        # 
        for i in range(ind_start,ind_end):
            junk_string = split_first(self.junk_se[i])
            if key == junk_string[0]:
                value = value + ' ; '+junk_string[1]
            else:
                key = junk_string[0]
                value = junk_string[1]
            header[key] = value
        return header
    
    def list_data(self,index = 'filename'):
        data = []
        for x in list(self.info.keys())[3:]:
            data.append(self.info[x][index])
        return data
    
    def plot_all(self,path = ''):
    
        keys = list(self.info.keys())[3:]
        for x in keys:
            filename = self.info[x]['filename']
            datafile = self.path +'%s' % (filename)
            b = Read_Mcstas(datafile)
            plt.figure()
            b.plot()
            plt.title(filename)

  
#======================================================================================
class Read_Mcstas():
    def __init__(self,filename):
        self.filename = filename
        self.read_header()
        self.read_data()

    def read_header(self):
        header = {}
        f = open(self.filename,'r')
        for line in f:
            if '# ' in line:
                if '# Data' in line:
                    break
                name = line.split(':')[0].strip().split(' ')[1]
                header[name] = line.split(':')[-1].strip()
        f.close()
        header['data_dimension'] = eval(header['type'].split('_')[1][0])
        if header['data_dimension']  == 2:
            bin_x = header['type'].split(',')[0].split('(')[1].strip()
            bin_y = header['type'].split(',')[1].split(')')[0].strip()
            header['bins'] = [int(bin_x),int(bin_y)]
            junk = [float(x) for x in header['xylimits'].split(' ')]
            header['xlimit'] = junk[:2]
            header['ylimit'] = junk[2:]
            header['step_x'] = (header['xlimit'][1] - header['xlimit'][0])/(header['bins'][0]-1)
            header['step_y'] = (header['ylimit'][1] - header['ylimit'][0])/(header['bins'][1]-1)
        elif header['data_dimension']  == 1:
            bin_x = header['type'].split('(')[1].split(')')[0].strip()
            header['bins'] = [int(bin_x)]
            header['xlimit'] = [float(x) for x in header['xlimits'].split(' ')]
        header['variables'] = header['variables'].split(' ')
        #header['date'] = header['Directory'].split('/')[-1].split('_')[-2]
        #header['id'] = header['Directory'].split('/')[-1].split('_')[-1]
        self.header = header

    def read_data(self):
        if self.header['data_dimension'] == 2:
            columns_n = self.header['bins'][0]
            self.data = pd.read_csv(self.filename,sep='\s+',comment='#',index_col=None,names=np.linspace(0,columns_n-1,columns_n))
        elif self.header['data_dimension']  == 1:
            self.data = pd.read_csv(self.filename,sep='\s+',comment='#',index_col=None,names=self.header['variables'])
    
    def select_data(self,variable = 0):
        self.variable = variable
        self.data_select = self.data.iloc[self.header['bins'][1]*variable:self.header['bins'][1]*(variable+1)].values
        self.mx,self.my = np.mgrid[slice(self.header['xlimit'][0],self.header['xlimit'][1]+self.header['step_x'],self.header['step_x']),
                       slice(self.header['ylimit'][0],self.header['ylimit'][1]+self.header['step_y'],self.header['step_y'])]
        self.x = self.mx.transpose()[0]
        self.y = self.my[0]
  
    def plot_2d(self,set_clim = -1,variable = 0,log = False,figsize=None):
        self.select_data(variable=variable)
        self.data_plot = self.data_select
        if log:
            self.data_plot = np.log(self.data_plot)
            minvalue = np.min(self.data_plot)
            self.data_plot[np.isinf(self.data_plot)] = minvalue
        # set figure size
        if figsize:
            plt.figure(figsize=(figsize[0],figsize[1]))
        plt.pcolormesh(self.mx,self.my,self.data_plot.transpose(),cmap = 'jet')
        cbar = plt.colorbar()
        cbar.set_label(self.header['variables'][variable])
        if set_clim != -1:
            cbar.set_clim(*set_clim)
        plt.xlabel(self.header['xlabel'])
        plt.ylabel(self.header['ylabel'])
        #title_text = '%s_%s.%s' %(self.header['date'],self.header['id'],self.header['filename'])
        #plt.title(title_text)
    
    def plot_1d(self,xran=False,**kargs):
        self.data_plot = self.data
        index_x = self.header['variables'][0]
        if xran:
            self.data_plot = self.data[self.data[index_x] >xran[0]][self.data[index_x] <xran[1]]
        x = self.data_plot[index_x]
        y = self.data_plot['I']
        dy = self.data_plot['I_err']
        plt.errorbar(x,y,yerr=dy,fmt='o-',**kargs)
        plt.xlabel(index_x)
        plt.ylabel('I')
    
    def plot(self):
        if self.header['data_dimension'] == 2:
            self.plot_2d()
        elif self.header['data_dimension']  == 1:
            self.plot_1d()
        
class Read_Mcstas_nd(Read_Mcstas):
    def __init__(self,filename):
        name_str = 'p x y z vx vy vz t sx sy sz I'
        self.columns_name = name_str.split()
        Read_Mcstas.__init__(self,filename)
        
    def read_data(self):
        self.data = pd.read_csv(self.filename,sep='\s+',comment='#',index_col=None,names=self.columns_name)
        self.data['div_x'] = self.data['vx']*180/self.data['vz']/np.pi
        self.data['div_y'] = self.data['vy']*180/self.data['vz']/np.pi
        self.data['v2'] = self.data['vx']**2 + self.data['vy']**2 + self.data['vz']**2
        self.data['e'] = m_value*self.data['v2']/2.0/meV   # meV
        self.data['lamb'] = 9.045/np.sqrt(self.data['e'])
    
    def plot_2d(self,ind = ['x','y'],bins_value=[20,20],weights = 'N'):
        if weights == 'N':
            plt.hist2d(self.data[ind[0]],self.data[ind[1]],bins=bins_value,cmap='jet')
        else:
            plt.hist2d(self.data[ind[0]],self.data[ind[1]],bins=bins_value,cmap='jet',weights=self.data[weights])
        plt.xlabel(ind[0])
        plt.ylabel(ind[1])
        plt.colorbar().set_label(weights)
    
    def plot_1d(self,ind='x',bins_value=20,weights = 'I'):
        # p也处理成I
        self.data_hist = pd.DataFrame()
        data = np.histogram(self.data[ind],bins=bins_value,weights=self.data[weights])
        self.data_hist[ind] = (data[1][1:]+data[1][:-1])/2
        self.data_hist['I'] = data[0]
        data = np.histogram(self.data[ind],bins=bins_value)
        self.data_hist['N'] = data[0]
        self.data_hist['I_err'] = self.data_hist['I']/np.sqrt(self.data_hist['N'])
        plt.plot(self.data_hist[ind],self.data_hist['I'],'o-')
        plt.xlabel(ind)
        plt.ylabel('I')

#======================================================================================
def lambda_to_e(lamb):
    return (9.045/lamb)**2

class Data_Slice():
    # 不负责xy调换，提前调换好再输入，默认都是x轴切片
    def __init__(self,data,xlist,ylist,index_x,index_y,name='I'):
        self.data = data   # matrix data, np.array
        self.xx = pd.Series(xlist,name = index_x)
        self.yy = pd.Series(ylist,name = index_y)
        self.bins = self.data.shape  # bins
        self.name = name
        
    def select_slice(self,x_select,plot=True):
        # 
        self.id_select = self.xx[self.xx<=x_select].idxmax()
        self.x_select = self.xx[self.id_select]
        
        df = pd.DataFrame(self.yy)
        df[self.name] = self.data[self.id_select]
        self.data_select = df
        if plot:
            self.plot()
    
    def plot(self):
        label_text = '%s = %.3f' % (self.xx.name,self.x_select)
        plt.plot(self.data_select[self.yy.name],self.data_select[self.name],label = label_text)
        plt.xlabel(self.yy.name)
        plt.ylabel(self.name)
        plt.legend()
        
class Data_Slice_McStas(Data_Slice):
    def __init__(self,rd,variable=0,direc = 'x'):
        # variable : 0 I / 1 I_err / 2 N
        # 输入的是Read_Mcstas object
        # 输出的应该是全部variable
        self.rd = rd
        rd.select_data(variable=variable)
        self.name = rd.header['variables'][variable]
        
        if direc == 'x':
            self.index_x = rd.header['xvar']
            self.index_y = rd.header['yvar']
            Data_Slice.__init__(self,rd.data_select.transpose(),rd.x,rd.y,self.index_x,
                                self.index_y,name = self.name)
    
        else:
            self.index_x = rd.header['yvar']
            self.index_y = rd.header['xvar']
            Data_Slice.__init__(self,rd.data_select,rd.y,rd.x,self.index_x,
                                self.index_y,name = self.name)

class Data_Slice_McStas_All():
    def __init__(self,rd):
        self.rd = rd
    
    def select_slice(self,value_select,direc='x'):
        df = pd.DataFrame()
        for i in range(3):
            junk = Data_Slice_McStas(self.rd,variable=i,direc = direc)
            junk.select_slice(value_select,plot=False)
            if len(df) == 0:
                df = junk.data_select
            else:
                df = pd.merge(df,junk.data_select)
        self.id_select = junk.id_select 
        self.x_select = junk.x_select
        self.index_x= junk.index_y
        self.data_select = df
        
class Peak_FWHM():
    '''
    计算mcstas计算结果的real fwhm和gaussian fwhm
        input : dataframe
        	columns : I,N,index_x
    '''

    def __init__(self,data,index_x):
        self.data = data
        self.index_x = index_x        
        
    def do_all_default(self,**kargs):
        self.measure_real_fwhm(**kargs)
        self.measure_integral_intensity()
        self.measure_gaussian_fwhm()
        self.plot()
        plt.legend()

    def measure_real_fwhm(self,index_y = 'I',bg = [],bg_points=[10,10]):
        # 计算波形的real fwhm
        # bg is the xrange for background estimate
        data = self.data.copy()
        if len(bg) == 0:
            self.data_bg = data.iloc[-1*bg_points[1]:].append(data.iloc[:bg_points[0]])
        else:
            self.data_bg = data[data[self.index_x]>=bg[0]][data[self.index_x]<=bg[1]]

        self.bg_value = self.data_bg[index_y].mean()
        # peak
        self.peak_ind = data[index_y].idxmax()
        self.peak_posi = data[self.index_x].loc[self.peak_ind]
        peak_y = data[index_y].loc[self.peak_ind]
        self.half_peak_value = (peak_y - self.bg_value)/2.0 + self.bg_value
        # 峰值左右分别插值
        self.x_left = select_interp(self.half_peak_value,data.loc[:self.peak_ind],index_x=index_y,index_y=self.index_x)
        self.x_right = select_interp(self.half_peak_value,data.loc[self.peak_ind:],index_x=index_y,index_y=self.index_x)
        self.real_fwhm = np.abs(self.x_right-self.x_left)
    
    def measure_gaussian_fwhm(self,index_y = 'N'):
        # 计算波形的gaussian equivalent fwhm
        xx = self.data[self.index_x].values
        yy = self.data[index_y].values
        points_list = []
        for j in range(len(xx)):
            points_list.extend(list(np.ones(int(yy[j]))*xx[j]))
        std = np.std(points_list)
        # 求std-FWHM 系数 A
        A = np.sqrt(8*np.log(2))
        self.gaussian_fwhm = std*A
    
    def measure_integral_intensity(self,index_y='I'):
        # substract bg
        yy = self.data[index_y].values - self.bg_value
        yy[yy<0] = 0
        self.dx = np.abs(self.data[self.index_x].iloc[1] - self.data[self.index_x].iloc[0])
        # width of x bin
        self.II = yy.sum()*self.dx
    
    def plot(self,xran=False,index_y = 'I',index_dy = 'I_err',label_text='',**kargs):
        if xran:
            self.data_plot = self.data[self.data[self.index_x] >xran[0]][self.data[self.index_x] <xran[1]]
        else:
            self.data_plot = self.data
        x = self.data_plot[self.index_x]
        y = self.data_plot[index_y]
        dy = self.data_plot[index_dy]
        if len(label_text) == 0:
        	label_text = 'g_fwhm: %.3e\nr_fwhm: %.3e\nII: %.3e' % (self.gaussian_fwhm,self.real_fwhm,self.II)
        plt.errorbar(x,y,yerr=dy,fmt='+-',label =label_text, **kargs)
        # real fwhm
        plt.plot([self.x_left,self.x_right],[self.half_peak_value]*2,'-',color='red',label = 'real FWHM')
        # bg
        plt.plot(x,[self.bg_value]*len(x),'--',color='yellow',label = 'bg line')
        plt.plot(self.data_bg[self.index_x],self.data_bg[index_y],'o',color='black',label='bg points')
        plt.xlabel(self.index_x)
        plt.ylabel(index_y)

#======================================================================================      
class Fermi_Mcstas():
    '''
    mcstas FermiChopper:
    ---------------------------------------------------------------------------------------------------
    radius = 0.1, nu = FC_Hz, xwidth = width_FC, yheight = height_FC, nslit=Nslit_FC, length=length_FC,
    verbose=1, phase=92.5767 )
    ---------------------------------------------------------------------------------------------------
      double w = width_FC/Nslit_FC;
      double v = 3956/lambda;

      printf("\nTheor: Lambda=%g [Angs]\n",
        3956*w/2/PI/FC_Hz/length_FC/length_FC/2);

      printf("Theor: Time from source  t=%g [s]\n", d_SF/v);
      printf("Theor: Time to detection t=%g [s]\n", d_FD/v);
      printf("       Time period      dt=%g [s]\n", 1/FC_Hz);
      printf("       Slit pack div       %g [deg] (full width)\n", 2*atan2(w,length_FC)/PI*180);
      printf("       Time window width  =%g [s] (pulse width)\n",
        atan2(w,length_FC)/PI/FC_Hz);
      printf("       Phase           phi=%g [deg]\n", (d_SF/v)/(1/FC_Hz)*360);

      time_to_arrival  = (d_SF/v);
      time_window_width= atan2(w,length_FC)/PI/FC_Hz;
      if (phase == -0) phase=(d_SF/v)/(1/FC_Hz)*360; /* assumes time at source is centered on 0 */
    %}
    '''
    def __init__(self,se):
        # se : FC_Hz ,width_FC,height_FC,Nslit_FC,length_FC
        self.para = se
        self.para['w'] = self.para['width_FC']/self.para['Nslit_FC']
        # chopper转速决定的cutoff lambda
        self.para['cut_off_lamb'] = 3956*self.para['w']/2/np.pi/self.para['FC_Hz']/self.para['length_FC']/self.para['length_FC']/2

    def calc_theory(self,en=1000,d_SF = 16):
        # en : 目标通过能量
        # d_SF : Fermi chopper距离moderator的长度 m
        self.para['d_SF'] = d_SF  # m
        theory = pd.Series()
        theory['en_set'] = en    # meV
        theory['lamb_set'] = 9.045/np.sqrt(theory['en_set'])  # A
        theory['v_set'] = 3956/theory['lamb_set']  # m/s
        theory['t_set'] = self.para['d_SF']/theory['v_set'] # s
        theory['phase_set'] = self.para['d_SF']/theory['v_set']/(1/self.para['FC_Hz'])*360 # deg
        self.theory = theory
        #self.para['time_window_width']= atan2(w,length_FC)/PI/FC_Hz

from bash import bash
class McStas_Run():
    # 特别提醒，lambda只能用sed改
    def __init__(self,filepath):
        self.path = filepath
        c = f'mkdir -p %s' % (self.path)
        bash(c)
    
    def cp_instr(self,origin_instr):
        self.file_instr = self.path + origin_instr.split('/')[-1]
        c = f'cp %s %s' % (origin_instr,self.path)
        bash(c)
    
    def read_instr(self):
        c = f'mcrun -i %s' % (self.file_instr)
        info = bash(c)
        info = info.value().split('\n')
        self.params = {}
        for x in info:
            if 'Param:' in x:
                junk = x.split(':')[-1].strip()
                key = junk.split('=')[0]
                values = junk.split('=')[1]
                self.params[key] = values
    
    def modify_instr(self,sed_str,sed_str_replace):
        nstr = len(sed_str)
        for i in range(nstr):
            c = f'sed -i -e "s#%s#%s#g" %s' %(sed_str[i],sed_str_replace[i],self.file_instr)
            bash(c)

    def run_instr(self,n_cpu = 1,counts='1E6',dirname='test',**kargs):
        def creat_kargs_string(**kargs):
            string = ''
            for x in kargs:
                string = string +'%s=%f ' % (x,kargs[x])
            return string
        kargs_string = creat_kargs_string(**kargs)
        self.dirname = self.path+dirname
        c = f'mcrun -c --mpi=%i %s -n %s -d %s %s' %(n_cpu,self.file_instr,counts,self.dirname,kargs_string)
        self.log = bash(c)
    
    def plot_all(self):
        simfile = self.dirname+'/mccode.sim'
        c = Read_Simu(simfile)
        keys = list(c.info.keys())[3:]
        for x in keys:
            filename = c.info[x]['filename']
            datafile = self.dirname+'/%s' % (filename)
            b = Read_Mcstas(datafile)
            plt.figure()
            b.plot()
