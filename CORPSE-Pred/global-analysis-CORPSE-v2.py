
def run_array(SOM_init,params,nyears,forcing,inputs,claydata,do_RK=False,output_yrs=1,Tref_decomp=293.15,Tref_predator=293.15):

    import xarray
    from numpy import zeros,asarray,arange,stack

    import time
    import CORPSE_array

    def rungekutta(state,func,dt,*args,**kwargs):
        class math_dict(dict):
            def __add__(self,val):
                if isinstance(val,dict):
                    out=self.copy()
                    for k in out.keys():
                        out[k]+=val[k]
                    return out
                else:
                    raise ValueError('Only add other dicts')
            def __mul__(self,val):
                if is_numlike(val):
                    out=self.copy()
                    for k in out.keys():
                        out[k]*=val
                    return out
                else:
                    raise ValueError('Only multiply by numbers')
            def __rmul__(self,val):
                return self*val
            def __truediv__(self,val):
                if is_numlike(val):
                    out=self.copy()
                    for k in out.keys():
                        out[k]/=val
                    return out
                else:
                    raise ValueError('Only divide by numbers')
            def copy(self):
                return math_dict(dict(self).copy())

        state_a=math_dict(state)
        k1=math_dict(func(state,*args,**kwargs))
        k2=math_dict(func(state_a+k1/2.0*dt,*args,**kwargs))
        k3=math_dict(func(state_a+k2/2.0*dt,*args,**kwargs))
        k4=math_dict(func(state_a+k3*dt,*args,**kwargs))
        return dict(state_a+dt/6.0*(k1+2*k2+2*k3+k4))

    dt=5.0/365
    t0=time.time()
    nsteps=int(nyears/dt)
    nlats=len(forcing['Ts']['lat'])
    nlons=len(forcing['Ts']['lon'])

    def read_init_cond(SOM_init):
        if isinstance(SOM_init,str):
            print('Reading initial SOM conditions from netCDF dataset %s'%SOM_init)
            SOM=read_init_cond(xarray.open_dataset(SOM_init))
        elif isinstance(SOM_init,xarray.Dataset):
            print('Reading initial SOM conditions from xarray dataset')
            SOM={}
            for f in SOM_init.data_vars:
                SOM[f]=zeros((nlats,nlons))
                SOM[f][:,:]+=SOM_init[f].values
        elif isinstance(SOM_init,dict):
            if 'nopred' in SOM_init.keys():
                pred=read_init_cond(SOM_init['pred'])
                nopred=read_init_cond(SOM_init['nopred'])
                SOM={}
                for f in pred.keys():
                    SOM[f]=zeros((nlats,nlons))
                    SOM[f][:,:]=pred[f][:,:]
            elif SOM_init['uFastC'].size==1:
                print('Initializing SOM conditions from cold start numbers')
                SOM={}
                for f in SOM_init.keys():
                    SOM[f]=zeros((nlats,nlons))
                    SOM[f][:,:]+=SOM_init[f]
            else:
                print('Using initial SOM conditions in dict format')
                SOM={}
                for f in SOM_init.keys():
                    SOM[f]=zeros((nlats,nlons))
                    SOM[f][:,:]+=SOM_init[f]
        else:
            raise ValueError('SOM_init in format %s not implemented'%str(type(SOM_init)))
        return SOM

    SOM=read_init_cond(SOM_init)

    SOM_out_accum=SOM.copy()
    SOM_out={}
    for f in SOM.keys():
        SOM_out[f]=zeros((len(inputs['Fast'].lat),len(inputs['Fast'].lon),nyears//output_yrs))

    # clay=stack((claydata.values,claydata.values),axis=-1)
    # claymod=CORPSE_array.prot_clay(clay)/CORPSE_array.prot_clay(20)
    #
    # Ts=stack((forcing['Ts'].values,forcing['Ts'].values),axis=-1)
    # Theta=stack((forcing['Theta'].values,forcing['Theta'].values),axis=-1)
    #
    # inputs_fast=stack((inputs['Fast'].values,inputs['Fast'].values),axis=-1)
    # inputs_slow=stack((inputs['Slow'].values,inputs['Slow'].values),axis=-1)
    clay=claydata.values
    claymod=CORPSE_array.prot_clay(clay)/CORPSE_array.prot_clay(20)
    Ts=forcing['Ts'].values
    Theta=forcing['Theta'].values
    inputs_fast=inputs['Fast'].values
    inputs_fast[inputs_fast<0.0]=0.0
    inputs_slow=inputs['Slow'].values
    inputs_slow[inputs_slow<0.0]=0.0

    t1=t0
    nsteps_year=floor(1/dt)
    for step in range(nsteps):
        if step%(nsteps_year*output_yrs)==0:
            tcurr=time.time()
            if step>0:
                timeleft=(nsteps-step)*(tcurr-t0)/step
                if timeleft>60:
                    print ('Year %d of %d. Time elapsed: %1.1fs. Time per year: %1.1fs. Est. remaining time: %1.1f min'%(step/nsteps_year,nsteps/nsteps_year,tcurr-t0,(tcurr-t1)/output_yrs,timeleft/60))
                else:
                    print ('Year %d of %d. Time elapsed: %1.1fs. Time per year: %1.1fs. Est. remaining time: %1.1fs'%(step/nsteps_year,nsteps/nsteps_year,tcurr-t0,(tcurr-t1)/output_yrs,timeleft))
            t1=tcurr
            for pool in SOM.keys():
                # This needs to be averaged, otherwise we are only saving one point in seasonal cycle
                SOM_out[pool][:,:,int(step/(nsteps_year*output_yrs))]=SOM_out_accum[pool]/(nsteps_year*output_yrs)
                SOM_out_accum[pool]=0.0
        if len(Ts.shape)==3:
            # Forcing is changing over time. Assume monthly resolution
            nsteps_month=nsteps_year//12
            forcing_ind=int(step/nsteps_month)%Ts.shape[0]
            Ts_now=Ts[forcing_ind,...]
            Theta_now=Theta[forcing_ind,...]
            inputs_fast_now=inputs_fast[forcing_ind,...]
            inputs_slow_now=inputs_slow[forcing_ind,...]
        else:
            Ts_now=Ts
            Theta_now=Theta
            inputs_fast_now=inputs_fast
            inputs_slow_now=inputs_slow
        if do_RK:
            RK=rungekutta(SOM,CORPSE_array.CORPSE_deriv,dt,Ts_now,Theta_now,params,claymod=claymod,Tref_decomp=Tref_decomp,Tref_predator=Tref_predator)
            for pool in SOM.keys():
                SOM[pool]=RK[pool]
        else:
            derivs=CORPSE_array.CORPSE_deriv(SOM,Ts_now,Theta_now,params,claymod=claymod,Tref_decomp=Tref_decomp,Tref_predator=Tref_predator)
            # if any(derivs['predatorC']>1e10):
            #     ind=nonzero(derivs['predatorC']>1e10)
            #     print(ind)
            #     print(derivs['predatorC'][ind],Ts_now[ind],Theta_now[ind],inputs_fast_now[ind],inputs_slow_now[ind])
            #     print(SOM['uFastC'][ind],SOM['livingMicrobeC'][ind])
            for pool in SOM.keys():
                SOM[pool]=SOM[pool]+derivs[pool]*dt
        SOM['uFastC']=SOM['uFastC']+inputs_fast_now*dt
        SOM['uSlowC']=SOM['uSlowC']+inputs_slow_now*dt
        for pool in SOM.keys():
            # This needs to be averaged, otherwise we are only saving one point in seasonal cycle
            SOM_out_accum[pool]+=SOM[pool]

    t1=time.time()
    print('Total time: %1.1f s'%(t1-t0))
    print('Time per timestep: %1.2g s'%((t1-t0)/nsteps))

    SOM_ds=xarray.Dataset(coords={'lon':inputs.lon,'lat':inputs.lat,'time':arange(nsteps//(nsteps_year*output_yrs))})
    # SOM_ds_nopred=xarray.Dataset(coords={'lon':inputs.lon,'lat':inputs.lat,'time':arange(nsteps//(nsteps_year*output_yrs))})
    for pool in SOM_out.keys():
        SOM_ds[pool]=(('lat','lon','time'),SOM_out[pool][:,:,:])
        # SOM_ds_nopred[pool]=(('lat','lon','time'),SOM_out[pool][:,:,:,1])
    return SOM_ds

def run_ODEsolver(SOM_init,params,times,forcing,inputs,claydata=None):
    from numpy import zeros,asarray,arange
    import CORPSE_array


    fields=SOM_init.keys()
    def odewrapper(SOM_list,t,T,theta,inputs_fast,inputs_slow,clay):
        SOM_dict={}
        for n in xrange(len(fields)):
            SOM_dict[fields[n]]=asarray(SOM_list[n])
        deriv=CORPSE_array.CORPSE_deriv(SOM_dict,T,theta,params,claymod=CORPSE_array.prot_clay(clay)/CORPSE_array.prot_clay(20),Tref_decomp=Tref_decomp,Tref_predator=Tref_predator)
        deriv['uFastC']=deriv['uFastC']+atleast_1d(inputs_fast)
        deriv['uSlowC']=deriv['uSlowC']+atleast_1d(inputs_slow)
        deriv['CO2']=0.0 # So other fields can be minimized. CO2 will grow if there are inputs
        vals=[deriv[f] for f in fields]

        return vals

    SOM_out=xarray.Dataset(coords={'lat':inputs.lat,'lon':inputs.lon,'time':times})
    nlons=len(SOM_out['lon'])
    nlats=len(SOM_out['lat'])
    for f in fields:
        SOM_out[f]=xarray.DataArray(zeros((nlats,nlons,len(times)))+nan,coords=[SOM_out['lat'],SOM_out['lon'],SOM_out['time']])
    SOM_out['num_iterations']=xarray.DataArray(zeros_like(forcing['Ts']),coords=[SOM_out['lat'],SOM_out['lon']])

    clay=claydata.fillna(20).values
    Ts=forcing['Ts'].values
    Theta=forcing['Theta'].values
    fast_in=inputs['Fast'].values
    slow_in=inputs['Slow'].values

    dt=1.0/365

    from scipy.integrate import odeint
    initvals=[SOM_init[f] for f in fields]

    import time
    t0=time.time()
    ndone=0
    nfev_done=0
    t1=time.time()
    for lon in range(nlons):
        for lat in range(nlats):
            if isfinite(forcing['Theta'].isel(lat=lat,lon=lon)):
                if ndone%10==0:
                    print ('Point %d of %d: lat=%1.1f,lon=%1.1f, mean nfev=%1.1f, time per point = %1.1g s'%(ndone,forcing['Theta'].count(),forcing.lat[lat],forcing.lon[lon],nfev_done/10.0,(time.time()-t1)/10))
                    nfev_done=0
                    t1=time.time()
                ndone+=1
                result,infodict=odeint(odewrapper,initvals,times,full_output=True,
                            args=(Ts[lat,lon],
                                  Theta[lat,lon],
                                  fast_in[lat,lon],slow_in[lat,lon],clay[lat,lon]))
                if infodict['message']!='Integration successful.':
                    print (infodict['message'])
                # print result,infodict
                for n in xrange(len(fields)):
                    SOM_out[fields[n]][{'lat':lat,'lon':lon}] =result[:,n]
                SOM_out['num_iterations'][{'lat':lat,'lon':lon}]=infodict['nfe'][-1]
                nfev_done=nfev_done+infodict['nfe'][-1]

            else:
                continue

    print ('Total time: %1.1 minutes'%((time.time()-t0)/60))

    return SOM_out




def find_equil(SOM_init,params,forcing,inputs,claydata=None):
    from numpy import asarray,atleast_1d
    import CORPSE_array
    fields=SOM_init.keys()

    def minwrapper(SOM_list,T,theta,inputs_fast,inputs_slow,clay):
        SOM_dict={}
        for n in xrange(len(fields)):
            SOM_dict[fields[n]]=asarray(SOM_list[n])
        deriv=CORPSE_array.CORPSE_deriv(SOM_dict,T,theta,params,claymod=CORPSE_array.prot_clay(clay)/CORPSE_array.prot_clay(20))
        deriv['uFastC']=deriv['uFastC']+atleast_1d(inputs_fast)
        deriv['uSlowC']=deriv['uSlowC']+atleast_1d(inputs_slow)
        deriv['CO2']=0.0 # So other fields can be minimized. CO2 will grow if there are inputs
        vals=[deriv[f] for f in fields]

        return vals

    from scipy.optimize import fsolve

    SOM_out=xarray.Dataset(coords=inputs.coords)
    nlons=len(SOM_out['lon'])
    nlats=len(SOM_out['lat'])
    for f in fields:
        SOM_out[f]=xarray.DataArray(zeros_like(forcing['Ts'])+nan,coords=[SOM_out['lat'],SOM_out['lon']])
    SOM_out['num_iterations']=xarray.DataArray(zeros_like(forcing['Ts']),coords=[SOM_out['lat'],SOM_out['lon']])
    SOM_out['fsolve_status']=xarray.DataArray(zeros_like(forcing['Ts'])+nan,coords=[SOM_out['lat'],SOM_out['lon']])

    clay=claydata.fillna(20).values
    Ts=forcing['Ts'].values
    Theta=forcing['Theta'].values
    fast_in=inputs['Fast'].values
    slow_in=inputs['Slow'].values


    import time
    t0=time.time()
    ndone=0
    nfev_done=0
    t1=time.time()
    for lon in xrange(nlons):
        for lat in xrange(nlats):
            if isfinite(forcing['Theta'].isel(lat=lat,lon=lon)):
                if ndone%10==0:
                    print ('Point %d of %d: lat=%1.1f,lon=%1.1f, mean nfev=%1.1f, time per point = %1.1g s'%(ndone,forcing['Theta'].count(),forcing.lat[lat],forcing.lon[lon],nfev_done/10.0,(time.time()-t1)/10))
                    nfev_done=0
                    t1=time.time()
                ndone+=1
                # Set things up so it can use a map of initial conditions instead of fixed values
                if isinstance(SOM_init,dict):
                    initvals=[SOM_init[f] for f in fields]
                elif isinstance(SOM_init,xarray.Dataset):
                    xx=SOM_init.isel(lat=lat,lon=lon)
                    initvals=[xx[f].values for f in fields]
                result,infodict,ier,mesg=fsolve(minwrapper,initvals,full_output=True,
                            args=(Ts[lat,lon],
                                  Theta[lat,lon],
                                  fast_in[lat,lon],slow_in[lat,lon],clay[lat,lon]))
                for n in xrange(len(fields)):
                    SOM_out[fields[n]][{'lat':lat,'lon':lon}] =result[n]
                SOM_out['num_iterations'][{'lat':lat,'lon':lon}]=infodict['nfev']
                nfev_done=nfev_done+infodict['nfev']
                SOM_out['fsolve_status'][{'lat':lat,'lon':lon}]=ier
                if ier!=1:
                    print ('Problem with point lat=%1.1f,lon=%1.1f: nfev=%d., totalC=%1.1g, Theta=%1.1f, Ts=%1.1f \n   %s'%(forcing.lat[lat],forcing.lon[lon],infodict['nfev'],result.sum(),Theta[lat,lon],Ts[lat,lon],mesg))
            else:
                continue

    print ('Total time: %1.1f s'%(time.time()-t0))

    return SOM_out

def totalCarbon(SOM):
    return SOM['pFastC']+SOM['pSlowC']+SOM['pNecroC']+\
           SOM['uFastC']+SOM['uSlowC']+SOM['uNecroC']+\
           SOM['predatorC']+SOM['livingMicrobeC']


def plot_map(data,cmap='wieder'):

    from cartopy.util import add_cyclic_point

    from matplotlib.colors import LinearSegmentedColormap
    levs=array([0.0,0.1,1,2,3,5,10,20,30,50,100,200,500])
    wieder_colormap=['#CCCCCC','#9460B3','#3F007D','#1600C8','#0000CB','#116D44','#46C31B','#CAF425','#FED924','#FD9C1D','#FC3F14','#FB1012']
    mapnorm=matplotlib.colors.BoundaryNorm(levs,len(levs)-1,clip=True)
    # set_cmap('plasma')
    wieder_cmap=(matplotlib.colors.ListedColormap(wieder_colormap,name='wieder_colormap'))

    if cmap == 'wieder':
        register_cmap(cmap=wieder_cmap)
        set_cmap('wieder_colormap')
    else:
        set_cmap(cmap)
        mapnorm=matplotlib.colors.Normalize()

    mapdata,lon=add_cyclic_point(data.values,data.lon)
    ax=gca()
    ax.coastlines()
    ax.pcolormesh(lon,data.lat,mapdata,norm=mapnorm)
    colorbar()


def plot_equils(equil,cmap='wieder'):
    # equil=equil_out.roll(lon=144/2)
    # lon=equil.lon.values.copy()
    # lon[lon>=180]=lon[lon>=180]-360

    from cartopy.util import add_cyclic_point

    from matplotlib.colors import LinearSegmentedColormap
    levs=array([0.0,0.1,1,2,3,5,10,20,30,50,100,200,500])
    wieder_colormap=['#CCCCCC','#9460B3','#3F007D','#1600C8','#0000CB','#116D44','#46C31B','#CAF425','#FED924','#FD9C1D','#FC3F14','#FB1012']
    # set_cmap('plasma')
    wieder_cmap=(matplotlib.colors.ListedColormap(wieder_colormap,name='wieder_colormap'))

    if cmap == 'wieder':
        register_cmap(cmap=wieder_cmap)
        mapnorm=matplotlib.colors.BoundaryNorm(levs,len(levs)-1,clip=True)
        set_cmap('wieder_colormap')
    else:
        set_cmap(cmap)
        mapnorm=matplotlib.colors.Normalize()

    totalC=ma.masked_invalid(add_cyclic_point(totalCarbon(equil).values))
    from CORPSE_array import sumCtypes
    unprotectedC=ma.masked_invalid(add_cyclic_point(sumCtypes(equil,'u').values))
    protectedC=ma.masked_invalid(add_cyclic_point(sumCtypes(equil,'p').values))

    xx,lon=add_cyclic_point(totalCarbon(equil).values,equil.lon)

    import cartopy.crs as ccrs
    ax=subplot(311,projection=ccrs.PlateCarree())
    title('Total C')
    gca().coastlines()
    h=ax.pcolormesh(lon,equil.lat,totalC,norm=mapnorm);colorbar(h)

    ax=subplot(312,projection=ccrs.PlateCarree())
    title('Unprotected C')
    gca().coastlines()
    h=ax.pcolormesh(lon,equil.lat,unprotectedC,norm=mapnorm);colorbar(h)

    ax=subplot(313,projection=ccrs.PlateCarree())
    title('Protected C')
    gca().coastlines()
    h=ax.pcolormesh(lon,equil.lat,protectedC,norm=mapnorm);colorbar(h)

    # subplot(224,projection=ccrs.PlateCarree())
    # title('Protected C fraction')
    # gca().coastlines()
    # contourf(lon,equil.lat,protectedC/(protectedC+unprotectedC),levels=arange(0,1.1,0.1),cmap=get_cmap('RdBu'));colorbar()

    tight_layout()


def apparent_Ea(T_cold,T_warm,C_cold,C_warm):
    kB=8.62e-5 #eV/K
    return kB*log(C_warm/C_cold)*1.0/(1/T_warm-1/T_cold)

if __name__ == '__main__':
    import xarray
    from pylab import *

# Note: conversion from J/mol (V*C/mol) to eV is (1/1.602e-19 e/C) * 1/6.02e23 (e/mol) = 1.037e-5
# This is the same as the ratio between the ideal gas constant R=8.314472 J/K/mol and the Bolzmann constant 8.62e-5 eV/K
    params={
        'vmaxref':{'Fast':9.0,'Slow':0.25,'Necro':4.5}, #  Relative maximum enzymatic decomp rates (year-1)
        'Ea':{'Fast':5e3,'Slow':30e3,'Necro':5e3},    # Activation energy (controls T dependence)
        'kC':{'Fast':0.01,'Slow':0.01,'Necro':0.01},    # Michaelis-Menton half saturation parameter (g microbial biomass/g substrate)
        'gas_diffusion_exp':0.6,  # Determines suppression of decomp at high soil moisture
        'substrate_diffusion_exp':1.5,   # Controls suppression of decomp at low soil moisture
        'minMicrobeC':1e-3,       # Minimum microbial biomass (fraction of total C)
        'Tmic':0.25,              # Microbial lifetime (years)
        'et':0.6,                 # Fraction of microbial biomass turnover that goes to necromass instead of to CO2
        'eup':{'Fast':0.6,'Slow':0.05,'Necro':0.6},     # Microbial carbon use efficiency for each substrate type (fast, slow, necromass)
        'tProtected':75.0,        # Protected C turnover time (years)
        'protection_rate':{'Fast':0.3,'Slow':0.001,'Necro':1.5}, # Protected carbon formation rate (year-1). Modify this for different soil textures
        'new_resp_units':True,
        'vmaxref_predator':4.0,
        'Ea_predator':30e3,
        'minPredatorC':0.001,
        'Tpredator':0.5,
        'et_predator':0.6,
        'eup_predator':0.5,
        'kC_predator':0.5,
    }


    SOM_init={'CO2': array(0.0),
     'livingMicrobeC': array(0.06082741340918269),
     'pFastC': array(1.9782703596751834),
     'pNecroC': array(22.14449924234682),
     'pSlowC': array(0.6191970075466381),
     'predatorC': array(0.037210950358798935),
     'uFastC': array(0.08792312709667481),
     'uNecroC': array(0.1968399932653051),
     'uSlowC': array(8.255960100621841)}

    SOM_init_nopred={'CO2': array(0.0),
     'livingMicrobeC': array(0.052619678096324576),
     'pFastC': array(2.484823946504599),
     'pNecroC': array(16.546772332246295),
     'pSlowC': array(0.7777478666811644),
     'predatorC': array(0.0),
     'uFastC': array(0.11043661984464885),
     'uNecroC': array(0.1470824207310782),
     'uSlowC': array(10.369971555748858)}

    # ghcn_temp=xarray.open_dataset('air.mon.mean.nc')
    LM3_output=xarray.open_dataset('lm3_output.nc')
    LM3_landstatic=xarray.open_dataset('land_static.nc')

    # Run with actual time series? We have monthly means here, probably better
    soilT_mean=LM3_output['tsoil_av'].mean(dim='time')
    theta_mean=LM3_output['theta'].mean(dim='time')
    npp_mean=LM3_output['npp'].mean(dim='time')
    soiltype=LM3_landstatic['soil_type']
    # inputs=xarray.Dataset({'Fast':npp_mean*0.3,'Slow':npp_mean*0.7,'Necro':npp_mean*0.0})
    npp=LM3_output['npp']
    inputs=xarray.Dataset({'Fast':npp*0.3,'Slow':npp*0.7,'Necro':npp*0.0})

    # From LM3 namelist parameters, matches to soil type (indexed from 1)
    clay  =  array([80.0  ,  50.0  ,  50.0  ,  40.0 ,   25.0,    11.0,    6.0,     45.0,    17.5,    27.5,    5.0 ,    10.0,    2.0,     17.5])
    clayarray=soiltype.values.copy()
    for val in range(len(clay)):
        clayarray[soiltype.values==val+1]=clay[val]
    claymap=xarray.DataArray(clayarray,dims=('lat','lon'),coords=(soiltype.lat,soiltype.lon))



    # forcing=xarray.Dataset({'Ts':soilT_mean,'Theta':theta_mean})
    forcing=xarray.Dataset({'Ts':LM3_output['tsoil_av'],'Theta':LM3_output['theta']})
    # equil_out=find_equil(SOM_init,params,forcing,inputs,claymap)
    times=array([0,1,10,100,1000,5000])
    nyears=500
    forcing_warmed=forcing.copy()
    forcing_warmed['Ts']=forcing_warmed['Ts']+2.0

    params_nopred=params.copy()
    params_nopred['Tmic']=0.25
    params_nopred['vmaxref_predator']=0.0
    # nopred=run_array('CORPSE_nopred_1000y_monthlyforcing.nc',params_nopred,nyears,forcing,inputs,claymap,do_RK=False,output_yrs=5)
    nopred=run_array('CORPSE_nopred_750years.nc',params_nopred,nyears,forcing,inputs,claymap,do_RK=False,output_yrs=5)
    nopred_warmed=run_array('CORPSE_nopred_750years.nc',params_nopred,nyears,forcing_warmed,inputs,claymap,do_RK=False,output_yrs=5)


    params['Tmic']=0.25
    Tref_pred=forcing['Ts'].mean(dim='time').clip(min=273.15)
    # pred=run_array('CORPSE_pred_1000y_monthlyforcing.nc',params,nyears,forcing,inputs,claymap,do_RK=False,output_yrs=5)
    pred_Ea30=run_array('CORPSE_pred_Ea30_051418.nc',params,nyears,forcing,inputs,claymap,do_RK=False,output_yrs=5,Tref_predator=Tref_pred)
    pred_Ea30_warmed=run_array('CORPSE_pred_Ea30_051418.nc',params,nyears,forcing_warmed,inputs,claymap,do_RK=False,output_yrs=5,Tref_predator=Tref_pred)
    
    pred_Ea30_constTref=run_array('CORPSE_pred_Ea30_051418.nc',params,nyears,forcing,inputs,claymap,do_RK=False,output_yrs=5,Tref_predator=float(Tref_pred.mean()))
    pred_Ea30_constTref_warmed=run_array('CORPSE_pred_Ea30_051418.nc',params,nyears,forcing_warmed,inputs,claymap,do_RK=False,output_yrs=5,Tref_predator=float(Tref_pred.mean()))

    params_pred_Ea10=params.copy()
    # params_pred_Ea10['vmaxref_predator']=3.0
    # params_pred_Eafixed['Ea']['Slow']=35e3
    params_pred_Ea10['Ea_predator']=10.0e3
    # params_pred_Ea10['Tmic']=0.4
    # pred_Eafixed=run_array('CORPSE_pred_1000y_monthlyforcing.nc',params_pred_Eafixed,nyears,forcing,inputs,claymap,do_RK=False,output_yrs=5)
    pred_Ea10=run_array('CORPSE_pred_Ea10_051418.nc',params_pred_Ea10,nyears,forcing,inputs,claymap,do_RK=False,output_yrs=5,Tref_predator=Tref_pred)
    pred_Ea10_warmed=run_array('CORPSE_pred_Ea10_051418.nc',params_pred_Ea10,nyears,forcing_warmed,inputs,claymap,do_RK=False,output_yrs=5,Tref_predator=Tref_pred)
    # ode_out=run_ODEsolver(SOM_init,params,times,forcing,inputs,claymap)

    # Try warming all three simulations: Plot warmed with predators minus warmed without predators
    # Global map of Q10?

    # Think about temperature optima: Talk about it in the Discussion. Also different baselines in different climates
    # Viruses should have same temperature sensitivity as microbes. Does this suggest that viruses should be more dominant
    # in high lats and predators more in warm climates?
    
    cell_area=LM3_landstatic['area_soil']
    totalC_nopred=(totalCarbon(nopred)*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    totalC_predEa30=(totalCarbon(pred_Ea30)*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    totalC_predEa10=(totalCarbon(pred_Ea10)*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    totalC_nopred_warmed=(totalCarbon(nopred_warmed)*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    totalC_predEa30_warmed=(totalCarbon(pred_Ea30_warmed)*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    totalC_predEa10_warmed=(totalCarbon(pred_Ea10_warmed)*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    totalC_predEa30_constTref=(totalCarbon(pred_Ea30_constTref)*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    totalC_predEa30_constTref_warmed=(totalCarbon(pred_Ea30_constTref_warmed)*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
        
    from CORPSE_array import sumCtypes
    unprotC_nopred=(sumCtypes(nopred,'u')*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    unprotC_predEa30=(sumCtypes(pred_Ea30,'u')*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    unprotC_predEa10=(sumCtypes(pred_Ea10,'u')*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    unprotC_nopred_warmed=(sumCtypes(nopred_warmed,'u')*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    unprotC_predEa30_warmed=(sumCtypes(pred_Ea30_warmed,'u')*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    unprotC_predEa10_warmed=(sumCtypes(pred_Ea10_warmed,'u')*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    unprotC_predEa30_constTref=(sumCtypes(pred_Ea30_constTref,'u')*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    unprotC_predEa30_constTref_warmed=(sumCtypes(pred_Ea30_constTref_warmed,'u')*cell_area).sum(skipna=True,dim=('lat','lon'))*1e-12
    t=arange(1,nyears+1,5)

    figure('Global average time series');clf()
    subplot(211)
    plot(t,1-totalC_nopred_warmed/totalC_nopred,label='No predators')
    plot(t,1-totalC_predEa30_warmed/totalC_predEa30,label='Warming responsive predators') 
    # plot(t,1-totalC_predEa10_warmed/totalC_predEa10,label='Warming insensitive predators') 
    plot(t,1-totalC_predEa30_constTref_warmed/totalC_predEa30_constTref,label='Constant Tref') 
    plot(t,1-unprotC_nopred_warmed/unprotC_nopred,ls='--',c='C0')
    plot(t,1-unprotC_predEa30_warmed/unprotC_predEa30,ls='--',c='C1') 
    # plot(t,1-unprotC_predEa10_warmed/unprotC_predEa10,ls='--',c='C2') 
    plot(t,1-unprotC_predEa30_constTref_warmed/unprotC_predEa30_constTref,ls='--',c='C2') 
    ylabel('Fractional global C loss')
    xlabel('Time (years)')
    title('Fractional global C loss')
    
    
    subplot(212)
    T_warm=forcing_warmed['Ts'].mean(dim='time')
    T_cold=forcing['Ts'].mean(dim='time')
    plot(t,apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=totalC_nopred_warmed,C_cold=totalC_nopred).mean(dim=('lat','lon')),label='No predators')
    plot(t,apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=totalC_predEa30_warmed,C_cold=totalC_predEa30).mean(dim=('lat','lon')),label='Warming responsive predators')
    plot(t,apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=totalC_predEa30_constTref_warmed,C_cold=totalC_predEa30_constTref).mean(dim=('lat','lon')),label='Constant Tref')
    # plot(t,apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=totalC_predEa10_warmed,C_cold=totalC_predEa10).mean(dim=('lat','lon')),label='Warming insensitive predators')
    plot(t,apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=unprotC_nopred_warmed,C_cold=unprotC_nopred).mean(dim=('lat','lon')),ls='--',c='C0')
    plot(t,apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=unprotC_predEa30_warmed,C_cold=unprotC_predEa30).mean(dim=('lat','lon')),ls='--',c='C1')
    plot(t,apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=unprotC_predEa30_constTref_warmed,C_cold=unprotC_predEa30_constTref).mean(dim=('lat','lon')),ls='--',c='C2')
    # plot(t,apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=unprotC_predEa10_warmed,C_cold=unprotC_predEa10).mean(dim=('lat','lon')),ls='--',c='C2')
    plot([t[0],t[-1]],[params['Ea']['Slow']*1.037e-5,params['Ea']['Slow']*1.037e-5],'k--',label='Slow C Ea parameter')
    title('Apparent Ea')
    xlabel('Time (years)')
    ylabel('Apparent Ea (eV)')
    legend()
    
    tight_layout()

    ygrids=[-40,-20,0,20,40,60]
    xgrids=None
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    # plot_equils(pred_warmed.isel(time=-1)-pred.isel(time=-1))
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    from CORPSE_array import sumCtypes
    from string import ascii_lowercase

    def letter_label(ax=None,xpos=-0.07,ypos=1.08,letter=None):
        if ax is None:
            ax=gca()
        from string import ascii_lowercase
        if letter is None:
            plotnum=ax.get_subplotspec().num1
            letter=ascii_lowercase[plotnum]
        return text(xpos,ypos,'('+letter+')',transform=ax.transAxes)


    figure('Apparent Ea',figsize=(8,8));clf()
    
    comparison_time = int(150/5)-1

    ax=subplot(311,projection=ccrs.PlateCarree())
    # mapdata,lon=add_cyclic_point(totalCarbon(pred_Ea30.isel(time=-1)).values-totalCarbon(nopred.isel(time=-1)).values,pred_Ea30.lon)
    Ea_apparent_nopred=apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=sumCtypes(nopred_warmed,'u'),C_cold=sumCtypes(nopred,'u')).isel(time=comparison_time).values
    mapdata,lon=add_cyclic_point(Ea_apparent_nopred,pred_Ea30.lon)
    
    levs=arange(-.75,.76,0.05)
    cmap='BrBG_r'
    
    # mapdata,lon=add_cyclic_point((totalCarbon(pred_Ea30_warmed)/totalCarbon(pred_Ea30)).isel(time=-1).values,pred_Ea30.lon)
    contourf(lon,nopred.lat,mapdata,cmap=cmap,levels=levs,extend='both')
    gca().coastlines()
    cb=colorbar()
    cb.set_label('Apparent Ea (eV)')
    title('No predators')
    letter_label(letter='a')
    gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    gl.ylabels_right=False
    gl.xlines=False
    gl.xlabels_top=False;gl.xlabels_bottom=False
    gl.yformatter = LATITUDE_FORMATTER

    ax=subplot(312,projection=ccrs.PlateCarree())
    # mapdata,lon=add_cyclic_point(totalCarbon(pred_Ea30.isel(time=-1)).values-totalCarbon(nopred.isel(time=-1)).values,pred_Ea30.lon)
    mapdata,lon=add_cyclic_point(apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=sumCtypes(pred_Ea30_warmed,'u'),C_cold=sumCtypes(pred_Ea30,'u')).isel(time=comparison_time).values,pred_Ea30.lon)
    # mapdata,lon=add_cyclic_point((totalCarbon(pred_Ea30_warmed)/totalCarbon(pred_Ea30)).isel(time=-1).values,pred_Ea30.lon)
    contourf(lon,nopred.lat,mapdata,cmap=cmap,levels=levs,extend='both')
    gca().coastlines()
    cb=colorbar()
    cb.set_label('Apparent Ea (eV)')
    title('Predators with locally adapted $T_{0}$')
    letter_label(letter='b')
    gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    gl.ylabels_right=False
    gl.xlines=False
    gl.xlabels_top=False;gl.xlabels_bottom=False
    gl.yformatter = LATITUDE_FORMATTER

    ax=subplot(313,projection=ccrs.PlateCarree())
    # mapdata=add_cyclic_point(totalCarbon(pred_Ea10.isel(time=-1)).values-totalCarbon(nopred.isel(time=-1)).values)
    # mapdata=add_cyclic_point(apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=sumCtypes(pred_Ea10_warmed,'u'),C_cold=sumCtypes(pred_Ea10,'u')).isel(time=-1).values)
    mapdata=add_cyclic_point(apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=sumCtypes(pred_Ea30_constTref_warmed,'u'),C_cold=sumCtypes(pred_Ea30_constTref,'u')).isel(time=comparison_time).values)
    # mapdata=add_cyclic_point((totalCarbon(pred_Ea30_warmed)/totalCarbon(pred_Ea30)).isel(time=-1).values)
    contourf(lon,nopred.lat,mapdata,cmap=cmap,levels=levs,extend='both')
    gca().coastlines()
    cb=colorbar()
    cb.set_label('Apparent Ea (eV)')
    # title('Low predator T response')
    title('Predators with globally constant $T_{0}$')
    letter_label(letter='c')
    gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    gl.ylabels_right=False
    gl.xlines=False
    gl.xlabels_top=False;gl.xlabels_bottom=False
    gl.yformatter = LATITUDE_FORMATTER
    
    tight_layout()

    def predfrac(data):
        return (data['predatorC'].isel(time=-1)/totalCarbon(data.isel(time=-1)))
        

    figure('Global maps',figsize=(12,5.5));clf()
    
    # 
    # ax=subplot(325,projection=ccrs.PlateCarree())
    # # mapdata,lon=add_cyclic_point(totalCarbon(pred_Ea30.isel(time=-1)).values-totalCarbon(nopred.isel(time=-1)).values,pred_Ea30.lon)
    # mapdata,lon=add_cyclic_point(apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=totalCarbon(pred_Ea30_warmed),C_cold=totalCarbon(pred_Ea30)).isel(time=-1).values-Ea_apparent_nopred,pred_Ea30.lon)
    # # mapdata,lon=add_cyclic_point((totalCarbon(pred_Ea30_warmed)/totalCarbon(pred_Ea30)).isel(time=-1).values,pred_Ea30.lon)
    # contourf(lon,nopred.lat,mapdata,cmap='BrBG_r',levels=arange(-0.5,0.51,0.02),extend='both')
    # gca().coastlines()
    # cb=colorbar()
    # cb.set_label('eV')
    # title('Apparent Ea difference')
    # letter_label(letter=ascii_lowercase[3-1])
    # gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    # gl.ylabels_right=False
    # gl.xlines=False
    # gl.xlabels_top=False;gl.xlabels_bottom=False
    # gl.yformatter = LATITUDE_FORMATTER
    # 
    # ax=subplot(326,projection=ccrs.PlateCarree())
    # # mapdata=add_cyclic_point(totalCarbon(pred_Ea10.isel(time=-1)).values-totalCarbon(nopred.isel(time=-1)).values)
    # mapdata=add_cyclic_point(apparent_Ea(T_warm=T_warm,T_cold=T_cold,C_warm=totalCarbon(pred_Ea30_constTref_warmed),C_cold=totalCarbon(pred_Ea30_constTref)).isel(time=-1).values-Ea_apparent_nopred)
    # # mapdata=add_cyclic_point((totalCarbon(pred_Ea30_warmed)/totalCarbon(pred_Ea30)).isel(time=-1).values)
    # contourf(lon,nopred.lat,mapdata,cmap='BrBG_r',levels=arange(-0.5,0.51,0.02),extend='both')
    # gca().coastlines()
    # cb=colorbar()
    # cb.set_label('eV')
    # title('Apparent Ea difference')
    # letter_label(letter=ascii_lowercase[3+3-1])
    # gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    # gl.ylabels_right=False
    # gl.xlines=False
    # gl.xlabels_top=False;gl.xlabels_bottom=False
    # gl.yformatter = LATITUDE_FORMATTER


    ax=subplot(221,projection=ccrs.PlateCarree())
    mapdata=add_cyclic_point(pred_Ea30['predatorC'].isel(time=-1).values)
    contourf(lon,nopred.lat,mapdata*1e3,cmap='magma_r',levels=arange(0,31,2.5)*2,extend='max')
    gca().coastlines()
    cb=colorbar()
    cb.set_label('g C m$^{-2}$')
    title('Locally adapted T$_{0}$:\nTotal predator biomass C')
    letter_label(letter=ascii_lowercase[1-1])
    gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    gl.ylabels_right=False
    gl.xlines=False
    gl.xlabels_top=False;gl.xlabels_bottom=False
    gl.yformatter = LATITUDE_FORMATTER

    ax=subplot(222,projection=ccrs.PlateCarree())
    mapdata=add_cyclic_point(pred_Ea30_constTref['predatorC'].isel(time=-1).values)
    contourf(lon,nopred.lat,mapdata*1e3,cmap='magma_r',levels=arange(0,31,2.5)*2,extend='max')
    gca().coastlines()
    cb=colorbar()
    cb.set_label('g C m$^{-2}$')
    title('Global constant T$_{0}$:\nTotal predator biomass C')
    letter_label(letter=ascii_lowercase[1+2-1])
    gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    gl.ylabels_right=False
    gl.xlines=False
    gl.xlabels_top=False;gl.xlabels_bottom=False
    gl.yformatter = LATITUDE_FORMATTER

    def micfrac(data):
        return (data['livingMicrobeC'].isel(time=-1)/totalCarbon(data.isel(time=-1)))

    # subplot(425,projection=ccrs.PlateCarree())
    # # mapdata=add_cyclic_point(pred_Ea30['livingMicrobeC'].isel(time=-1).values-nopred['livingMicrobeC'].isel(time=-1).values)
    # mapdata=add_cyclic_point(micfrac(pred_Ea30).values)*100
    # contourf(lon,nopred.lat,mapdata,cmap='YlGn',levels=arange(0,0.8,0.05),extend='max')
    # gca().coastlines()
    # # colorbar()
    # title('Microbial biomass fraction')
    #
    # subplot(426,projection=ccrs.PlateCarree())
    # # mapdata=add_cyclic_point(pred_Ea10['livingMicrobeC'].isel(time=-1).values-nopred['livingMicrobeC'].isel(time=-1).values)
    # mapdata=add_cyclic_point(micfrac(pred_Ea10).values)*100
    # contourf(lon,nopred.lat,mapdata,cmap='YlGn',levels=arange(0,0.8,0.05),extend='max')
    # gca().coastlines()
    # cb=colorbar()
    # cb.set_label('% of total C')
    # title('Microbial biomass fraction')

    ax=subplot(223,projection=ccrs.PlateCarree())
    mapdata=add_cyclic_point(micfrac(pred_Ea30).values/micfrac(nopred).values-1)*100
    contourf(lon,nopred.lat,mapdata,cmap='BrBG_r',levels=arange(-50,51,5)*2,extend='both')
    gca().coastlines()
    cb=colorbar()
    cb.set_label('% Difference from no-pred')
    title('% Difference in microbial biomass fraction')
    letter_label(letter=ascii_lowercase[2-1])
    gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    gl.ylabels_right=False
    gl.xlines=False
    gl.xlabels_top=False;gl.xlabels_bottom=False
    gl.yformatter = LATITUDE_FORMATTER

    ax=subplot(224,projection=ccrs.PlateCarree())
    mapdata=add_cyclic_point(micfrac(pred_Ea30_constTref).values/micfrac(nopred).values-1)*100
    contourf(lon,nopred.lat,mapdata,cmap='BrBG_r',levels=arange(-50,51,5)*2,extend='both')
    gca().coastlines()
    cb=colorbar()
    cb.set_label('% Difference from no-pred')
    title('% Difference in microbial biomass fraction')
    letter_label(letter=ascii_lowercase[2+2-1])
    gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    gl.ylabels_right=False
    gl.xlines=False
    gl.xlabels_top=False;gl.xlabels_bottom=False
    gl.yformatter = LATITUDE_FORMATTER

    tight_layout()





    cell_area=LM3_landstatic['area_soil']
    totalC_nopred=(totalCarbon(nopred)*cell_area).isel(time=20).sum(skipna=True,dim=('lat','lon'))*1e-12
    totalC_hiTsens=(totalCarbon(pred_Ea30)*cell_area).isel(time=20).sum(skipna=True,dim=('lat','lon'))*1e-12
    totalC_loTsens=(totalCarbon(pred_Ea10)*cell_area).isel(time=20).sum(skipna=True,dim=('lat','lon'))*1e-12
    
    print('Total SOC (No pred): %1.1f Pg'%totalC_nopred)
    print('Total SOC (High T sens pred): %1.1f Pg (%1.1f%% more than no-pred)'%(totalC_hiTsens,(totalC_hiTsens/totalC_nopred-1)*100))
    print('Total SOC (Low T sens pred): %1.1f Pg (%1.1f%% more than no-pred)'%(totalC_loTsens,(totalC_loTsens/totalC_nopred-1)*100))
    # 
    # #### Warming comparison ####
    # figure(2,figsize=(12,7));clf()
    # x_text=0.03
    # y_text=0.07
    # boxprops={'facecolor':'white','linewidth':0.5,'alpha':1.0}
    # fontsize=7
    # 
    # let_xpos=-0.09
    # units_text='SOC change\n(% of control SOC stock)'
    # 
    # ax=subplot(321,projection=ccrs.PlateCarree())
    # nopredloss=(totalCarbon(nopred_warmed.isel(time=20)).values/totalCarbon(nopred.isel(time=20)).values)-1
    # # mapdata,lon=add_cyclic_point(nopredloss/totalCarbon(nopred.isel(time=20)).values,pred_Ea30.lon)
    # mapdata=add_cyclic_point(nopredloss)
    # contourf(lon,nopred.lat,mapdata*100,cmap='BrBG_r',levels=arange(-12,12.5,2),extend='both')
    # gca().coastlines()
    # cb=colorbar();cb.set_label(units_text,fontsize='small')
    # title('SOC change (No predators)')
    # letter_label(letter='a',xpos=let_xpos)
    # gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    # gl.ylabels_right=False
    # gl.xlines=False
    # gl.xlabels_top=False;gl.xlabels_bottom=False
    # gl.yformatter = LATITUDE_FORMATTER
    # totalC_control=(totalCarbon(nopred)*cell_area).isel(time=20).sum(skipna=True,dim=('lat','lon'))*1e-12
    # totalC_warmed=(totalCarbon(nopred_warmed)*cell_area).isel(time=20).sum(skipna=True,dim=('lat','lon'))*1e-12
    # text(x_text,y_text,'C loss: %1.1f Pg\n(%1.1f%% of control)'%(totalC_control-totalC_warmed,(100-totalC_warmed/totalC_control*100)),
    #         transform=ax.transAxes,fontsize=fontsize,bbox=boxprops,va='bottom')
    # 
    # 
    # ax=subplot(323,projection=ccrs.PlateCarree())
    # pred30loss=(totalCarbon(pred_Ea30_warmed.isel(time=20)).values/totalCarbon(pred_Ea30.isel(time=20)).values)-1
    # # mapdata=add_cyclic_point(pred30loss/totalCarbon(pred_Ea30.isel(time=20)).values)
    # mapdata=add_cyclic_point(pred30loss)
    # contourf(lon,nopred.lat,mapdata*100,cmap='BrBG_r',levels=arange(-12,12.5,2),extend='both')
    # gca().coastlines()
    # cb=colorbar();cb.set_label(units_text,fontsize='small')
    # title('SOC change (High T sens. predators)')
    # letter_label(letter='b',xpos=let_xpos)
    # gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    # gl.ylabels_right=False
    # gl.xlines=False
    # gl.xlabels_top=False;gl.xlabels_bottom=False
    # gl.yformatter = LATITUDE_FORMATTER
    # totalC_control=(totalCarbon(pred_Ea30)*cell_area).isel(time=20).sum(skipna=True,dim=('lat','lon'))*1e-12
    # totalC_warmed=(totalCarbon(pred_Ea30_warmed)*cell_area).isel(time=20).sum(skipna=True,dim=('lat','lon'))*1e-12
    # text(x_text,y_text,'C loss: %1.1f Pg\n(%1.1f%% of control)'%(totalC_control-totalC_warmed,(100-totalC_warmed/totalC_control*100)),
    #         transform=ax.transAxes,fontsize=fontsize,bbox=boxprops,va='bottom')
    # 
    # ax=subplot(325,projection=ccrs.PlateCarree())
    # pred10loss=(totalCarbon(pred_Ea10_warmed.isel(time=20)).values/totalCarbon(pred_Ea10.isel(time=20)).values)-1
    # # mapdata=add_cyclic_point(pred10loss/totalCarbon(pred_Ea10.isel(time=20)).values)
    # mapdata=add_cyclic_point(pred10loss)
    # contourf(lon,nopred.lat,mapdata*100,cmap='BrBG_r',levels=arange(-12,12.5,2),extend='both')
    # gca().coastlines()
    # cb=colorbar();cb.set_label(units_text,fontsize='small')
    # title('SOC change (Low T sens. predators)')
    # letter_label(letter='c',xpos=let_xpos)
    # gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    # gl.ylabels_right=False
    # gl.xlines=False
    # gl.xlabels_top=False;gl.xlabels_bottom=False
    # gl.yformatter = LATITUDE_FORMATTER
    # totalC_control=(totalCarbon(pred_Ea10)*cell_area).isel(time=20).sum(skipna=True,dim=('lat','lon'))*1e-12
    # totalC_warmed=(totalCarbon(pred_Ea10_warmed)*cell_area).isel(time=20).sum(skipna=True,dim=('lat','lon'))*1e-12
    # text(x_text,y_text,'C loss: %1.1f Pg\n(%1.1f%% of control)'%(totalC_control-totalC_warmed,(100-totalC_warmed/totalC_control*100)),
    #         transform=ax.transAxes,fontsize=fontsize,bbox=boxprops,va='bottom')
    # 
    # 
    # ax=subplot(324,projection=ccrs.PlateCarree())
    # # mapdata=add_cyclic_point(pred30loss/totalCarbon(pred_Ea30.isel(time=20)).values)
    # mapdata=add_cyclic_point(pred30loss-nopredloss)
    # contourf(lon,nopred.lat,mapdata*100,cmap='BrBG_r',levels=arange(-12,12.5,2)*0.5,extend='both')
    # gca().coastlines()
    # cb=colorbar();cb.set_label('Difference from no-pred SOC loss\nMore SOC loss $\Longleftrightarrow$ Less SOC loss\n(% of control SOC stock)',fontsize='small')
    # title('Predator effect on SOC loss (High T sens.)')
    # letter_label(letter='e',xpos=let_xpos)
    # gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    # gl.ylabels_right=False
    # gl.xlines=False
    # gl.xlabels_top=False;gl.xlabels_bottom=False
    # gl.yformatter = LATITUDE_FORMATTER
    # 
    # ax=subplot(326,projection=ccrs.PlateCarree())
    # # pred10loss=(totalCarbon(pred_Ea10_warmed.isel(time=20)).values/totalCarbon(pred_Ea10.isel(time=20)).values)-1
    # # mapdata=add_cyclic_point(pred10loss/totalCarbon(pred_Ea10.isel(time=20)).values)
    # mapdata=add_cyclic_point(pred10loss-nopredloss)
    # contourf(lon,nopred.lat,mapdata*100,cmap='BrBG_r',levels=arange(-12,12.5,2)*0.5,extend='both')
    # gca().coastlines()
    # cb=colorbar();cb.set_label('Difference from no-pred SOC loss\nMore SOC loss $\Longleftrightarrow$ Less SOC loss\n(% of control SOC stock)',fontsize='small')
    # title('Predator effect on SOC loss (Low T sens.)')
    # letter_label(letter='f',xpos=let_xpos)
    # gl=ax.gridlines(draw_labels=True,ylocs=ygrids,xlocs=xgrids)
    # gl.ylabels_right=False
    # gl.xlines=False
    # gl.xlabels_top=False;gl.xlabels_bottom=False
    # gl.yformatter = LATITUDE_FORMATTER
    # 
    # ax=subplot(322)
    # lat=nopred.lat
    # plot(lat,((totalCarbon(nopred_warmed.isel(time=20))/totalCarbon(nopred.isel(time=20)))-1).mean(dim='lon')*100,label='No-pred')
    # plot(lat,((totalCarbon(pred_Ea30_warmed.isel(time=20))/totalCarbon(pred_Ea30.isel(time=20)))-1).mean(dim='lon')*100,label='High T sens.')
    # plot(lat,((totalCarbon(pred_Ea10_warmed.isel(time=20))/totalCarbon(pred_Ea10.isel(time=20)))-1).mean(dim='lon')*100,label='Low T sens.')
    # plot(lat,lat*0,'k:',lw=0.5,label='__nolabel__')
    # legend(fontsize='small',ncol=3)
    # title('Mean % change by latitude')
    # ylabel(units_text)
    # # ylabel('Latitude')
    # letter_label(letter='d',xpos=let_xpos)
    # xticks(ygrids)
    # xlim(-70,90)
    # ax.grid(True,axis='x')
    # ax.xaxis.set_major_formatter(LATITUDE_FORMATTER)
    # 
    # tight_layout()
    # subplots_adjust(left=0.05)
    # 



    figure('Zonal means and global means');clf()
    subplot(311)
    lat=pred_Ea30.lat
    cell_area=LM3_landstatic['area_soil']

    plot(lat,totalCarbon(pred_Ea30).mean(dim=('lon')).isel(time=-1),c='C0',ls='-',lw=1.0,label='Pred (Ea 30)')
    plot(lat,totalCarbon(pred_Ea30).mean(dim=('lon')).isel(time=1),c='C0',ls='--',lw=1.0)
    plot(lat,totalCarbon(nopred).mean(dim=('lon')).isel(time=-1),c='C1',ls='-',lw=1.0,label='No pred')
    plot(lat,totalCarbon(nopred).mean(dim=('lon')).isel(time=1),c='C1',ls='--',lw=1.0)
    plot(lat,totalCarbon(pred_Ea30_constTref).mean(dim=('lon')).isel(time=-1),c='C2',ls='-',lw=1.0,label='Pred (const Tref)')
    plot(lat,totalCarbon(pred_Ea30_constTref).mean(dim=('lon')).isel(time=1),c='C2',ls='--',lw=1.0)
    legend()
    subplot(312)
    for n,lt in enumerate([30,45,62,73]):

        plot(totalCarbon(pred_Ea30_warmed).isel(lat=lt).mean(dim='lon')[1:],c='C%d'%n,lw=1.0,label='Lat = %1.1f'%pred_Ea30.lat[lt])
        plot(totalCarbon(nopred_warmed).isel(lat=lt).mean(dim='lon')[1:],ls='--',c='C%d'%n,lw=1.0)
        plot(totalCarbon(pred_Ea30_constTref_warmed).isel(lat=lt).mean(dim='lon')[1:],ls=':',c='C%d'%n,lw=1.0)
    legend()
    subplot(313)
    plot((totalCarbon(pred_Ea30)*cell_area).sum(skipna=True,dim=('lat','lon'))[1:]*1e-12,c='C0',ls='-')
    plot((totalCarbon(nopred)*cell_area).sum(skipna=True,dim=('lat','lon'))[1:]*1e-12,c='C1',ls='-')
    plot((totalCarbon(pred_Ea30_constTref)*cell_area).sum(skipna=True,dim=('lat','lon'))[1:]*1e-12,c='C2',ls='-')
    plot((totalCarbon(pred_Ea30_warmed)*cell_area).sum(skipna=True,dim=('lat','lon'))[1:]*1e-12,c='C0',ls='--')
    plot((totalCarbon(nopred_warmed)*cell_area).sum(skipna=True,dim=('lat','lon'))[1:]*1e-12,c='C1',ls='--')
    plot((totalCarbon(pred_Ea30_constTref_warmed)*cell_area).sum(skipna=True,dim=('lat','lon'))[1:]*1e-12,c='C2',ls='--')

    tight_layout()

    figure('Stocks: No pred');clf()
    plot_equils(nopred.isel(time=-1))
    figure('Stocks: Ea=30');clf()
    plot_equils(pred_Ea30.isel(time=-1))
    figure('Stocks: Constant Tref');clf()
    plot_equils(pred_Ea30_constTref.isel(time=-1))

    show()
