from pylab import *      # importa matplotlib
from mpl_toolkits.axes_grid.axislines import * # importa i toolkits per avere anche i minor ticks nei plot
from scipy import stats
import numpy as np
import linmix

rc('text', usetex=True)
rc('font',family='Times New Roman')
rc('xtick', labelsize=13)
rc('ytick', labelsize=13)
fig = plt.figure(figsize=(8,8))

def func(x, a, b):
    '''linear 2-param function.'''
    return a + (b * x)

def predband(x, xd, yd, f_vars, conf=0.95):
    """
    Calculates the prediction band of the regression model at the
    desired confidence level.

    References:
    1. http://www.JerryDallal.com/LHSP/slr.htm, Introduction to Simple Linear
    Regression, Gerard E. Dallal, Ph.D.
    """

    alpha = 1. - conf    # Significance
    N = xd.size          # data sample size
    var_n = len(f_vars)  # Number of variables used by the fitted function.

    # Quantile of Student's t distribution for p=(1 - alpha/2)
    q = stats.t.ppf(1. - alpha / 2., N - var_n)

    # Std. deviation of an individual measurement (Bevington, eq. 6.15)
    se = np.sqrt(1. / (N - var_n) * np.sum((yd - func(xd, *f_vars)) ** 2))

    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)

    # Predicted values (best-fit model)
    yp = func(x, *f_vars)
    # Prediction band
    dy = q * se * np.sqrt(1. + (1. / N) + (sx / sxd))

    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy

    return lpb, upb

lcf = open('/Users/uramito/Dropbox/DES_paper/4OPS_data.txt','r')      # def di file-oggetto
riga1 = lcf.readlines()
lcf.close()

sn = 0
redshift = 1
Lp = 2
colpeak = 4
col30 = 6
d30 = 8

### format of test data
lcf = open('/Users/uramito/Dropbox/DES_paper/4OPS_testdata.txt','r')
rigat = lcf.readlines()
rigatest = rigat[1:]
lcf.close()
sn_name, sn_lpeak, sn_lpeak_err, sn_d30,sn_d30_err, sn_colpeak,sn_colpeak_err, sn_col30,sn_col30_err = [],[],[],[],[],[],[],[],[]
for line in rigatest:
  p = line.split()
  sn_d30.append(float(p[3]))
  sn_d30_err.append(float(p[4]))
  sn_lpeak.append(float(p[1]))
  sn_lpeak_err.append(float(p[2]))
  sn_colpeak.append(float(p[5]))
  sn_colpeak_err.append(float(p[6]))
  sn_col30.append(float(p[7]))
  sn_col30_err.append(float(p[8]))
  sn_name.append(p[0])

ax=fig.add_subplot(221)

yl = array([-16.7,-24.8])#   OKKKKK
xl = array([-0.2, 3.0])

X, Y, X_err, Y_err, snm = [],[],[],[],[]
for line in riga1:
  p = line.split()
  if float(p[d30]) != 9999:
      X.append(float(p[d30]))
      Y.append(float(p[Lp]))
      X_err.append(float(p[d30+1]))
      Y_err.append(float(p[Lp+1]))
      snm.append(p[sn])

Xfa = np.array(X)
Yfa = np.array(Y)
Xfa_err = np.array(X_err)
Yfa_err = np.array(Y_err)

# Bayeian weighted linear regression using monte carlo for random draw
lm = linmix.LinMix(Xfa, Yfa, xsig=Xfa_err, ysig=Yfa_err, K=3)
lm.run_mcmc(miniter=5000, maxiter=100000,silent=True)

### plotting the mean linear regreession and confidence bands
for i in xrange(0, len(lm.chain), 25):
    xs = np.arange(Xfa.min(),Xfa.max(),0.01)

ys = lm.chain['alpha'].mean() + xs * lm.chain['beta'].mean()
xp = linspace(-0.2,3.0,32)
ys1 = lm.chain['alpha'].mean() + xp * lm.chain['beta'].mean()
sigma = np.sqrt(lm.chain['sigsqr'].mean())
plot(xp, ys1,ls='--',color='#1f77b4',label='_nolegend_')
popt = (lm.chain['alpha'].mean(), lm.chain['beta'].mean())
low_l, up_l = predband(xp, Xfa, Yfa, popt, conf=0.996)

#### plotting data to test
print '\nPanel A\n'
i = 0
for sn in sn_d30:
    smb = ['*','h','>','s','o','d','^','D','^','v']
    color1 = ['k','purple','c','#d62728','g','b','#e377c2','gold','grey','lime']
    color2 = ['k','purple','c','#d62728','g','b','#e377c2','gold','grey','lime']
    size = [12,8,8,8,8,8,8,8,8,8,8,8]
    print 'Id:',sn_name[i], ' symbol:',smb[i]
    errorbar(sn_d30[i],sn_lpeak[i],xerr=sn_d30_err[i],yerr=sn_lpeak_err[i],marker='.',ms=8,color=color2[i],ecolor=color2[i],ls='None',label='_nolegend_')
    plot(sn_d30[i],sn_lpeak[i],marker=smb[i],ls='None',ms=size[i],color=color1[i],markeredgecolor=color2[i],label='_nolegend_')
    i = i +1

ax.fill_between(xp,up_l,low_l,where=None,alpha=0.2,facecolor='#1f77b4',edgecolor ='#1f77b4',zorder=-3, label='_nolegend_')
plot(xp, up_l,ls='-',color='#1f77b4',label='_nolegend_')
plot(xp, low_l,ls='-',color='#1f77b4',label='_nolegend_')

### axes plot
text(2.7,-24,'A',size=20)
ylabel(r'$M$(400)$_{\rm 0}$',fontsize=20)
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax.minorticks_on()
ax.set_xticklabels(['','','','','',''])

# create twin axis to have axes labels on top and right part of the plots
# quite old style, to be uppdated to lighten the code
ax2 = ax.twiny()
yl = array([-16.7,-24.8])
xl = array([-0.2, 3.0])
plot(9999,9999,'k.')
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax2.set_xlabel(r'${\rm \Delta}M$(400)$_{\rm 30}$',fontsize=20)
ax2.minorticks_on()

ax3 = ax.twinx()
yl = array([-16.7,-24.8])#   OKKKKK
xl = array([-0.2, 3.0])
plot(9999,9999,'k.')
ax3.set_yticklabels(['','','','','',''])
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax3.minorticks_on()


## second plot. Formalism of plot 2 to 4 is identycal to plot 1
ax=fig.add_subplot(222)

yl = array([-16.7,-24.8])
xl = array([-0.6, 1.3])

X, Y, X_err, Y_err = [],[],[],[]
for line in riga1:
  p = line.split()
  if float(p[col30]) != 9999:
      X.append(float(p[col30]))
      Y.append(float(p[Lp]))
      X_err.append(float(p[col30+1]))
      Y_err.append(float(p[Lp+1]))

Xfa = np.array(X)
Yfa = np.array(Y)
Xfa_err = np.array(X_err)
Yfa_err = np.array(Y_err)
lm = linmix.LinMix(Xfa, Yfa, xsig=Xfa_err, ysig=Yfa_err, K=3)
lm.run_mcmc(miniter=5000, maxiter=100000,silent=True)

for i in xrange(0, len(lm.chain), 25):
    xs = np.arange(Xfa.min(),Xfa.max(),0.01)

ys = lm.chain['alpha'].mean() + xs * lm.chain['beta'].mean()
xp = linspace(-0.6,1.3,17)
ys1 = lm.chain['alpha'].mean() + xp * lm.chain['beta'].mean()
sigma = np.sqrt(lm.chain['sigsqr'].mean())
plot(xp, ys1,ls='--',color='orange',label='_nolegend_')
popt = (lm.chain['alpha'].mean(), lm.chain['beta'].mean())
low_l, up_l = predband(xp, Xfa, Yfa, popt, conf=0.996)

print '\nPanel B\n'
i = 0
for sn in sn_col30:
    smb = ['*','h','>','s','o','d','^','D','^','v']
    color1 = ['k','purple','c','#d62728','g','b','#e377c2','gold','grey','lime']
    color2 = ['k','purple','c','#d62728','g','b','#e377c2','gold','grey','lime']
    size = [12,8,8,8,8,8,8,8,8,8,8,8]
    print 'Id:',sn_name[i], ' symbol:',smb[i]
    errorbar(sn_col30[i],sn_lpeak[i],xerr=sn_col30_err[i],yerr=sn_lpeak_err[i],marker='.',ms=8,color=color2[i],ecolor=color2[i],ls='None',label='_nolegend_')
    plot(sn_col30[i],sn_lpeak[i],marker=smb[i],ls='None',ms=size[i],color=color1[i],markeredgecolor=color2[i],label='_nolegend_')
    i = i +1

ax.fill_between(xp,up_l,low_l,where=None,alpha=0.2,facecolor='orange',edgecolor='orange',zorder=-3,label='3$\sigma$')
plot(xp, up_l,ls='-',color='orange',label='_nolegend_')
plot(xp, low_l,ls='-',color='orange',label='_nolegend_')

text(1.1,-24,'B',size=20)
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax.minorticks_on()
ax.set_xticklabels(['','','','','',''])
ax.set_yticklabels(['','','','','',''])

ax2 = ax.twiny()
yl = array([-16.7,-24.8])
xl = array([-0.6, 1.3])

plot(9999,9999,'k.')

xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax2.set_xlabel(r'$M$(400)$_{\rm 30}$ - $M$(520)$_{\rm 30}$',fontsize=20)
ax2.minorticks_on()

ax3 = ax.twinx()
yl = array([-16.7,-24.8])
xl = array([-0.6, 1.3])
plot(9999,9999,'k.')
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax3.set_ylabel(r'$M$(400)$_{\rm 0}$',fontsize=20)
ax3.minorticks_on()

ax=fig.add_subplot(223)
yl = array([-2.0,0.8])
xl = array([-0.2, 3.0])

X, Y, X_err, Y_err = [],[],[],[]
for line in riga1:
  p = line.split()
  if float(p[colpeak]) != 9999:
      X.append(float(p[d30]))
      Y.append(float(p[colpeak]))
      X_err.append(float(p[d30+1]))
      Y_err.append(float(p[colpeak+1]))

Xfa = np.array(X)
Yfa = np.array(Y)
Xfa_err = np.array(X_err)
Yfa_err = np.array(Y_err)
lm = linmix.LinMix(Xfa, Yfa, xsig=Xfa_err, ysig=Yfa_err, K=3)
lm.run_mcmc(miniter=5000, maxiter=100000,silent=True)

for i in xrange(0, len(lm.chain), 25):
    xs = np.arange(Xfa.min(),Xfa.max(),0.01)

ys = lm.chain['alpha'].mean() + xs * lm.chain['beta'].mean()

xp = linspace(-0.2,3.0,32)
ys1 = lm.chain['alpha'].mean() + xp * lm.chain['beta'].mean()
sigma = np.sqrt(lm.chain['sigsqr'].mean())
plot(xp, ys1,ls='--',color='orange',label='_nolegend_')
popt = (lm.chain['alpha'].mean(), lm.chain['beta'].mean())
low_l, up_l = predband(xp, Xfa, Yfa, popt, conf=0.996)

print '\nPanel C\n'
i = 0
for sn in sn_d30:
    smb = ['*','h','>','s','o','d','^','D','^','v']
    color1 = ['k','purple','c','#d62728','g','b','#e377c2','gold','grey','lime']
    color2 = ['k','purple','c','#d62728','g','b','#e377c2','gold','grey','lime']
    size = [12,8,8,8,8,8,8,8,8,8,8,8]
    print 'Id:',sn_name[i], ' symbol:',smb[i]
    errorbar(sn_d30[i],sn_colpeak[i],xerr=sn_d30_err[i],yerr=sn_colpeak_err[i],marker='.',ms=8,color=color2[i],ecolor=color2[i],ls='None',label='_nolegend_')
    plot(sn_d30[i],sn_colpeak[i],marker=smb[i],ls='None',ms=size[i],color=color1[i],markeredgecolor=color2[i],label='_nolegend_')
    i = i +1

ax.fill_between(xp,up_l,low_l,where=None,alpha=0.2,facecolor='orange',edgecolor='orange',zorder=-3, label='3$\sigma$')
plot(xp, up_l,ls='-',color='orange',label='_nolegend_')
plot(xp, low_l,ls='-',color='orange',label='_nolegend_')


text(0.15,0.6,'C',size=20)

xlabel(r'${\rm \Delta}M$(400)$_{\rm 30}$',fontsize=20)
ylabel(r'$M$(400)$_{\rm 0}$ - $M$(520)$_{\rm 0}$',fontsize=20)
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax.minorticks_on()

ax3 = ax.twinx()
yl = array([-2.0,0.8])
xl = array([-0.2, 3.0])
plot(9999,9999,'k.')
ax3.set_yticklabels(['','','','','',''])
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax3.minorticks_on()

ax2 = ax.twiny()
yl = array([-2.0,0.8])
xl = array([-0.2,3.0])
plot(9999,9999,'k.')
ax2.set_xticklabels(['','','','','',''])
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax2.minorticks_on()

ax=fig.add_subplot(224)
yl = array([-2.0,0.8])
xl = array([-0.6, 1.3])

X, Y, X_err, Y_err = [],[],[],[]
for line in riga1:
  p = line.split()
  if float(p[col30]) != 9999:
      X.append(float(p[col30]))
      Y.append(float(p[colpeak]))
      X_err.append(float(p[col30+1]))
      Y_err.append(float(p[colpeak+1]))


Xfa = np.array(X)
Yfa = np.array(Y)
Xfa_err = np.array(X_err)
Yfa_err = np.array(Y_err)

lm = linmix.LinMix(Xfa, Yfa, xsig=Xfa_err, ysig=Yfa_err, K=3)
lm.run_mcmc(miniter=5000, maxiter=100000,silent=True)


for i in xrange(0, len(lm.chain), 25):
    xs = np.arange(Xfa.min(),Xfa.max(),0.01)

ys = lm.chain['alpha'].mean() + xs * lm.chain['beta'].mean()
xp = linspace(-0.6,1.3,17)
ys1 = lm.chain['alpha'].mean() + xp * lm.chain['beta'].mean()
sigma = np.sqrt(lm.chain['sigsqr'].mean())
plot(xp, ys1,ls='--',label='_nolegend_')
popt = (lm.chain['alpha'].mean(), lm.chain['beta'].mean())
low_l, up_l = predband(xp, Xfa, Yfa, popt, conf=0.996)

print '\nPanel D\n'
i = 0
for sn in sn_col30:
    smb = ['*','h','>','s','o','d','^','D','^','v']
    color1 = ['k','purple','c','#d62728','g','b','#e377c2','gold','grey','lime']
    color2 = ['k','purple','c','#d62728','g','b','#e377c2','gold','grey','lime']
    size = [12,8,8,8,8,8,8,8,8,8,8,8]
    print 'Id:',sn_name[i], ' symbol:',smb[i]
    errorbar(sn_col30[i],sn_colpeak[i],xerr=sn_col30_err[i],yerr=sn_colpeak_err[i],marker='.',ms=8,color=color2[i],ecolor=color2[i],ls='None',label='_nolegend_')
    plot(sn_col30[i],sn_colpeak[i],marker=smb[i],ls='None',ms=size[i],color=color1[i],markeredgecolor=color2[i],label='_nolegend_')
    i = i +1

ax.fill_between(xp,up_l,low_l,where=None,alpha=0.2,zorder=-3,label='_nolegend_')
plot(xp, up_l,ls='-',color='#1f77b4',label='_nolegend_')
plot(xp, low_l,ls='-',color='#1f77b4',label='_nolegend_')
plt.legend(loc=4,ncol=1,prop={'size':9})
text(-0.35,0.6,'D',size=20)
xlabel(r'$M$(400)$_{\rm 30}$ - $M$(520)$_{\rm 30}$',fontsize=20)
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax.minorticks_on()
ax.set_yticklabels(['','','','','',''])

ax3 = ax.twinx()
yl = array([-2.0,0.8])
xl = array([-0.6, 1.3])
plot(9999,9999,'k.')
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax3.set_ylabel(r'$M$(400)$_{\rm 0}$ - $M$(520)$_{\rm 0}$',fontsize=20)
ax3.minorticks_on()

ax2 = ax.twiny()
yl = array([-2.0,0.8])
xl = array([-0.6, 1.3])
plot(9999,9999,'k.')
ax2.set_xticklabels(['','',''])
xlim(xl[0],xl[1])
ylim(yl[0],yl[1])
ax2.minorticks_on()

fig.subplots_adjust(hspace=0.07,wspace=0.07,left=0.12,right=0.88,top=0.92,bottom=0.09)

show()

#fig.savefig('SLSNe_4OPS.pdf',bbox_inches='tight',format='pdf',dpi=1000)
