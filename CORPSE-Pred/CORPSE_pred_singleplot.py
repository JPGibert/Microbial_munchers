
from pylab import *
from CORPSE_array import sumCtypes


def run_ODEsolver(SOM_init,params,times,forcing,clay=20):
     from numpy import zeros,asarray,arange
     import CORPSE_array


     fields=list(SOM_init.keys())
     def odewrapper(SOM_list,t,T,theta,inputs_fast,inputs_slow,clay):
         SOM_dict={}
         for n in range(len(fields)):
             SOM_dict[fields[n]]=asarray(SOM_list[n])
         deriv=CORPSE_array.CORPSE_deriv(SOM_dict,T,theta,params,claymod=CORPSE_array.prot_clay(clay)/CORPSE_array.prot_clay(20))
         deriv['uFastC']=deriv['uFastC']+atleast_1d(inputs_fast)
         deriv['uSlowC']=deriv['uSlowC']+atleast_1d(inputs_slow)
         deriv['CO2']=0.0 # So other fields can be minimized. CO2 will grow if there are inputs
         vals=[deriv[f] for f in fields]

         return vals

     SOM_out={}

     for f in fields:
         SOM_out[f]=0

     Ts=forcing['Ts']
     Theta=forcing['Theta']
     fast_in=forcing['Input_fast']
     slow_in=forcing['Input_slow']

     from scipy.integrate import odeint
     initvals=[SOM_init[f] for f in fields]


     result,infodict=odeint(odewrapper,initvals,times,full_output=True,
                 args=(Ts,Theta,fast_in,slow_in,clay))
     if infodict['message']!='Integration successful.':
         print (infodict['message'])
     # print result,infodict
     for n in range(len(fields)):
         SOM_out[fields[n]] =result[:,n]

     return SOM_out

def plot_timeseries(t,outputs,do_legend=True,**kwargs):
    plot(t,outputs['uFastC'],c='C0',label='Fast',**kwargs)
    plot(t,outputs['uNecroC'],c='C1',label='Dead mic',**kwargs)
    plot(t,outputs['uSlowC'],c='C2',label='Slow',**kwargs)
    plot(t,outputs['livingMicrobeC'],c='C3',label='Live Mic',**kwargs)
    plot(t,outputs['predatorC'],c='C4',label='Predators',**kwargs)
    plot(t,outputs['pFastC']+outputs['pSlowC']+outputs['pNecroC'],c='C5',label='Protected',**kwargs)

    plot([t[0],t[-1]],[0,0],'k:',lw=0.5)

    if do_legend:
        legend(fontsize='small',ncol=2,loc='upper right')
    title('Small carbon pools')
    ylabel('Carbon pools (kgC/m$^2$)')
    xlabel('Time (years)')

    tight_layout()

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

SOM_init = {'CO2': 0.0,
 'livingMicrobeC': 0.06380681641247811,
 'pFastC': 2.5041836226718766,
 'pNecroC': 12.961643714595255,
 'pSlowC': 0.7839106102086612,
 'predatorC': 0.02406677517237494,
 'uFastC': 0.11112896767521879,
 'uNecroC': 0.11549817099607097,
 'uSlowC': 10.435206668185971}

SOM_init_nopred={'CO2': 0.0,
 'livingMicrobeC': 0.09112556684686321,
 'pFastC': 1.6694815470262971,
 'pNecroC': 11.21683127669391,
 'pSlowC': 0.522546361410256,
 'predatorC': 0.0,
 'uFastC': 0.07419918181553982,
 'uNecroC': 0.09970515935842993,
 'uSlowC': 6.967285000769504}

params_nopred=params.copy()
params_nopred['vmaxref_predator']=0.0
params_nopred['minPredatorC']=0.0


t=arange(0,2000,10)
out=run_ODEsolver(SOM_init,params,times=t,forcing={'Ts':290,'Theta':0.5,'Input_fast':0.3,'Input_slow':0.7})
out_nopred=run_ODEsolver(SOM_init_nopred,params_nopred,times=t,forcing={'Ts':290,'Theta':0.5,'Input_fast':0.3,'Input_slow':0.7})

figure('Timeseries');clf()
plot_timeseries(t,out)
plot_timeseries(t,out_nopred,ls='--',do_legend=False)


# Plot dependence of pools on total C inputs.
# Will not be very interesting because CORPSE structure guarantees linear response to total inputs
input_rates=linspace(0.01,1.0,15)

n=len(input_rates)
microbes=zeros(n)
microbes_nopred=zeros(n)
SOC=zeros(n)
SOC_nopred=zeros(n)
predators=zeros(n)
microbes_warmed=zeros(n)
microbes_nopred_warmed=zeros(n)
SOC_warmed=zeros(n)
SOC_nopred_warmed=zeros(n)
predators_warmed=zeros(n)
for num in range(n):
    print('Input rate = %1.2f'%input_rates[num])
    out=run_ODEsolver(SOM_init,params,times=t,forcing={'Ts':290,'Theta':0.5,'Input_fast':0.3*input_rates[num],'Input_slow':0.7*input_rates[num]})
    out_nopred=run_ODEsolver(SOM_init_nopred,params_nopred,times=t,forcing={'Ts':290,'Theta':0.5,'Input_fast':0.3*input_rates[num],'Input_slow':0.7*input_rates[num]})
    out_warmed=run_ODEsolver(SOM_init,params,times=t,forcing={'Ts':292,'Theta':0.5,'Input_fast':0.3*input_rates[num],'Input_slow':0.7*input_rates[num]})
    out_nopred_warmed=run_ODEsolver(SOM_init_nopred,params_nopred,times=t,forcing={'Ts':292,'Theta':0.5,'Input_fast':0.3*input_rates[num],'Input_slow':0.7*input_rates[num]})
    microbes[num]=out['livingMicrobeC'][-1]
    microbes_nopred[num]=out_nopred['livingMicrobeC'][-1]
    SOC[num]=sumCtypes(out,'u')[-1]+sumCtypes(out,'p')[-1]
    SOC_nopred[num]=sumCtypes(out_nopred,'u')[-1]+sumCtypes(out_nopred,'p')[-1]
    predators[num]=out['predatorC'][-1]
    microbes_warmed[num]=out_warmed['livingMicrobeC'][-1]
    microbes_nopred_warmed[num]=out_nopred_warmed['livingMicrobeC'][-1]
    SOC_warmed[num]=sumCtypes(out_warmed,'u')[-1]+sumCtypes(out_warmed,'p')[-1]
    SOC_nopred_warmed[num]=sumCtypes(out_nopred_warmed,'u')[-1]+sumCtypes(out_nopred_warmed,'p')[-1]
    predators_warmed[num]=out_warmed['predatorC'][-1]

figure('Input rate dependence',figsize=(8,4));clf()
subplot(221)
plot(input_rates,SOC_nopred,'k--',label='C')
plot(input_rates,SOC_nopred_warmed,'k-',label='C (warmed)')
xlabel('Substrate input rate (kgC m$^{-2}$ year $^{-1}$)')
ylabel('Steady state mass (kgC m$^{-2}$)')
title('A. Microbial model')
ylim(-0.7,28.5)

subplot(223)
plot(input_rates,microbes_nopred,ls='--',color='orange',label='M')
plot(input_rates,microbes_nopred_warmed,ls='-',color='orange',label='M (warmed)')
xlabel('Substrate input rate (kgC m$^{-2}$ year $^{-1}$)')
ylabel('Steady state mass (kgC m$^{-2}$)')
ylim(-0.002,0.095)

subplot(222)
plot(input_rates,SOC,'k--',label='C')
plot(input_rates,SOC_warmed,'k-',label='C (warmed)')
xlabel('Substrate input rate (kgC m$^{-2}$ year $^{-1}$)')
ylabel('Steady state mass (kgC m$^{-2}$)')
title('B. Microbe-predator model')
legend()
ylim(-0.7,28.5)

subplot(224)
plot(input_rates,microbes,ls='--',color='orange',label='M')
plot(input_rates,predators,ls='--',color='purple',label='P')
plot(input_rates,microbes_warmed,ls='-',color='orange',label='M (warmed)')
plot(input_rates,predators_warmed,ls='-',color='purple',label='P (warmed)')
xlabel('Substrate input rate (kgC m$^{-2}$ year $^{-1}$)')
ylabel('Steady state mass (kgC m$^{-2}$)')
# title('A. Microbe-predator model')
legend()
ylim(-0.002,0.095)

tight_layout()

# Different temperature sensitivities
n=20
Ea_pred=linspace(10e3,70e3,n)
warming=[0,1,3,5]
microbes={}
SOC={}
predators={}

for temp in warming:
    microbes[temp]=zeros(n)
    SOC[temp]=zeros(n)
    predators[temp]=zeros(n)


for num in range(n):
    params_Eapred=params.copy()
    params_Eapred['Ea_predator']=Ea_pred[num]
    print('Ea_pred = %1.2e'%Ea_pred[num])
    for temp in warming:
        out=run_ODEsolver(SOM_init,params_Eapred,times=t,forcing={'Ts':290+temp,'Theta':0.5,'Input_fast':0.3*0.5,'Input_slow':0.7*0.5})

        microbes[temp][num]=out['livingMicrobeC'][-1]
        SOC[temp][num]=sumCtypes(out,'u')[-1]+sumCtypes(out,'p')[-1]
        predators[temp][num]=out['predatorC'][-1]

figure('Ea dependence');clf()
cmap=get_cmap('YlOrRd')
for temp in warming[1:]:
    plot((Ea_pred-params['Ea']['Slow'])*1.037e-5,(SOC[temp]-SOC[0])/SOC[0]*100,label='+ %d C'%temp,c=cmap(temp/max(warming)))

xlims=xlim()
ylims=ylim()
plot(xlims,[0,0],'k:',lw=0.5)
plot([0,0],ylims,'k:',lw=0.5)
xlim(*xlims)
ylim(*ylims)

legend()
xlabel('Ea difference from predator to microbe (eV)')
ylabel('SOC difference (%)')
title('Effect of predator Ea on SOC warming response')

show()
