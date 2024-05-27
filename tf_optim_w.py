import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import net_params as pr
import time
import math
from copy import deepcopy

class OriginFiring:
    def __init__(self, params, dt=0.1):
        self.dt = dt

    def integrate(self, n_steps):
        pass

    def getCompartmentByName(self, name):
        return self

    def getV(self):
        return tf.constant(0.0, dtype=tf.float32)

    def setIext(self, Iext):
        pass

    def addIsyn(self, gsyn, gE):
        pass

    def get_firing(self):
        firing = tf.zeros([1], dtype=tf.float32)
        return firing


class VonMissesGenerator(OriginFiring):
    def __init__(self, params, dt=0.1):
        self.dt = dt
        params = params["params"]

        Rs = np.zeros(len(params), dtype=np.float32)
        omegas = np.zeros_like(Rs)
        self.phases = np.zeros_like(Rs)
        mean_spike_rates = np.zeros_like(Rs)

        for p_idx, params_el in enumerate(params):
            Rs[p_idx] = tf.constant(params_el["R"])
            omegas[p_idx] = params_el["freq"]
            self.phases[p_idx] = params_el["phase"]
            mean_spike_rates[p_idx] = params_el["mean_spike_rate"]

        self.kappa = self.r2kappa(Rs)
        pi = tf.constant(math.pi)

        self.mult4time = 2 * pi * omegas * 0.001

        I0 = tf.math.bessel_i0(self.kappa)
        self.normalizator = mean_spike_rates / I0 * 0.001 * self.dt  # units: probability of spikes during dt

        self.t = 0.0

    def r2kappa(self, R):
        """
        recalulate kappa from R for von Misses function
        """
        kappa = tf.where(R < 0.53,  2 * R + tf.pow(R, 3) + 5 / 6 * tf.pow(R, 5), 0.0)
        kappa = tf.where(tf.logical_and(R >= 0.53, R < 0.85),  -0.4 + 1.39 * R + 0.43 / (1 - R), kappa)
        kappa = tf.where(R >= 0.85,  1 / (3 * R - 4 * tf.pow(R, 2) + tf.pow(R, 3)), kappa)
        return kappa

    def get_firing(self, t=None):
        if t is None:
            t = self.t

        t = tf.convert_to_tensor(t, dtype=tf.float32)

        firings = self.normalizator * tf.exp(self.kappa * tf.cos(self.mult4time * t - self.phases))
        self.t += self.dt
        return firings

    def integrate(self, n_steps):
        self.t += n_steps * self.dt

class VonMissesSpatialModulation(VonMissesGenerator):
    def __init__(self, params, dt=0.1):
        self.dt = dt
        super().__init__(params, dt)
        params = params["params"]

        sigma_sp = tf.Variable(np.zeros(len(params), dtype=np.float32))
        maxFiring = tf.Variable(np.zeros_like(sigma_sp))
        v_an = tf.Variable(np.zeros_like(sigma_sp))
        mean_spike_rates = tf.Variable(np.zeros_like(sigma_sp))
        sp_centers = tf.Variable(np.zeros_like(sigma_sp))

        for p_idx, params_el in enumerate(params):
            sigma_sp[p_idx].assign(params_el["sigma_sp"])
            maxFiring[p_idx].assign(params_el["maxFiring"])
            v_an[p_idx].assign(0.001 * params_el["v_an"])
            mean_spike_rates[p_idx].assign(params_el["mean_spike_rate"])
            sp_centers[p_idx].assign(params_el["sp_centers"])

        self.sigma_t = sigma_sp / v_an
        self.t_centers = sp_centers / v_an
        self.Amps = 2 * (maxFiring - mean_spike_rates) / (mean_spike_rates + 1)

    def get_firing(self, t=None):
        if t is None:
            t = self.t

        t = tf.convert_to_tensor(t, dtype=tf.float32)

        phase_firings = self.normalizator * tf.exp(self.kappa * tf.cos(self.mult4time * t - self.phases))
        spatial_firings = 1 + self.Amps * tf.exp(-0.5 * tf.pow((t - self.t_centers) / self.sigma_t, 2))

        firings = phase_firings * spatial_firings
        return firings





#@tf.function
def simulate(X0 : tf.Variable, Duration, dt, Cm, animal_velocity, params_generators, params_synapses):
    print('params_synapses_["Erev"] 2 ', params_synapses["Erev"])
    params_generators_ = params_generators
    params_synapses_ = params_synapses
    x_idx = 0
    for gen in params_generators_["params"][1:]:
        gen["maxFiring"] = X0[x_idx]
        x_idx += 1

        gen["sp_centers"] = X0[x_idx] + tf.Variable(0.5 * 0.001 * Duration * animal_velocity)
        x_idx += 1

        gen["sigma_sp"] = X0[x_idx]
        x_idx += 1
    
    #print(params_generators["params"][1:])

    Gplus = X0[-1]
    print('params_synapses_["Erev"]', params_synapses_["Erev"])

    Erev = tf.cast(params_synapses_["Erev"], dtype=tf.float32) #cast -> constant
    Gmax = tf.cast(X0[x_idx:-1], dtype=tf.float32)

    tau_d = tf.constant(params_synapses_["tau_d"], dtype=tf.float32)
    tau_r = tf.constant(params_synapses_["tau_r"], dtype=tf.float32)
    tau_f = tf.constant(params_synapses_["tau_f"], dtype=tf.float32)
    Uinc = tf.constant(params_synapses_["Uinc"], dtype=tf.float32)
    pconn = tf.constant(params_synapses_["pconn"], dtype=tf.float32)

    generators = VonMissesSpatialModulation(params_generators_)#getattr(lib, params_generators["class"])(params_generators)

    #t = tf.range(0, Duration, dt, dtype=tf.float32)
    t = np.arange(0, Duration, dt)

    tau1r = tau_d / (tau_d - tau_r)

    R = tf.zeros_like(Gmax)
    X = tf.ones_like(Gmax)
    U = tf.zeros_like(Gmax)

    exp_tau_r = tf.exp(-dt / tau_r)
    exp_tau_d = tf.exp(-dt / tau_d)
    exp_tau_f = tf.exp(-dt / tau_f)

    g_hist = []

    for ts in t:
        SRpre = generators.get_firing(ts)
        Spre_normed = SRpre * pconn

        y_ = R * exp_tau_d
        x_ = 1 + (X - 1 + tau1r * U) * exp_tau_r - tau1r * U
        u_ = U * exp_tau_f

        released_mediator = U * x_ * Spre_normed
        U = u_ + Uinc * (1 - u_) * Spre_normed
        R = y_ + released_mediator
        X = x_ - released_mediator

        R_plus = tf.concat([R, [0.5]], axis=0)
        g_hist.append(R_plus)

    g_hist = tf.stack(g_hist)

    Gmax_plus = tf.concat([Gmax, [Gplus]], axis=0)
    g_hist_plus = Gmax_plus * g_hist

    G_tot_plus = tf.reduce_sum(g_hist_plus, axis=1)
    #print('Erev', Erev)
    Erev_plus = tf.concat([Erev, [60]], axis=0)
    Erev_plus = tf.reshape(Erev_plus, [1, -1])


    Erev_hist = tf.reduce_sum(g_hist_plus * Erev_plus, axis=1) / (G_tot_plus + 0.1)
    #tau_m_hist = Cm / (G_tot_plus + 0.1)

    return Erev_hist

def get_target_Esyn(t, tc, dt, theta_freq, v_an, params):
    ALPHA = 5.0
    # meanSR = params['mean_firing_rate']
    phase = np.deg2rad(params['phase_out_place'])
    kappa = 0.15  # self.r2kappa(params["R_place_cell"])
    maxFiring = 20  # params['peak_firing_rate']

    SLOPE = np.deg2rad(params['precession_slope'] * v_an * 0.001)  # rad / ms
    ONSET = np.deg2rad(params['precession_onset'])

    sigma_spt = params['sigma_place_field'] / v_an * 1000

    mult4time = 2 * np.pi * theta_freq * 0.001

    # I0 = bessel_i0(kappa)
    normalizator = 1 # 0.1  # meanSR / I0 * 0.001 * dt

    amp = 17  # 2 * (maxFiring - meanSR) / (meanSR + 1)  # maxFiring / meanSR - 1 #  range [-1, inf]

    # print(meanSR)
    multip = amp * np.exp(-0.5 * ((t - tc) / sigma_spt) ** 2)

    start_place = t - tc - 3 * sigma_spt
    end_place = t - tc + 3 * sigma_spt
    inplace = 0.25 * (1.0 - (start_place / (ALPHA + np.abs(start_place)))) * (
            1.0 + end_place / (ALPHA + np.abs(end_place)))

    precession = SLOPE * t * inplace
    phases = phase * (1 - inplace) - ONSET * inplace

    Esyn = 15.6 + multip + normalizator * np.cos(mult4time * t + precession - phases)

    return Esyn

animal_velocity = pr.V_AN
theta_freq = pr.theta_generators["params"][0]["freq"]
theta_generator = deepcopy(pr.theta_generators)
theta_generator["params"][0]["sigma_sp"] = 10000000
theta_generator["params"][0]["v_an"] = pr.V_AN
theta_generator["params"][0]["maxFiring"] = 24.0
theta_generator["params"][0]["sp_centers"] = 100000000

params_generators = {
    "class": "VonMissesSpatialMolulation",
    "name": "theta_spatial_inputs",
    "params" : [theta_generator["params"][0], ],
}


params_generators["params"].extend( deepcopy(pr.theta_spatial_generators_soma["params"]) )
params_generators["params"].extend( deepcopy(pr.theta_spatial_generators_dend["params"]) )

params_synapses = {}

Erev = []
Gmax = []
tau_d = [] #6.489890385
tau_r = [] # 801.5798994
tau_f = [] # 19.0939326
Uinc = []  #0.220334906
pconn = []  #0.220334906

for Syn in pr.synapses_params:
    for p in Syn['params']:
        Erev.append(p["Erev"])
        Gmax.append(p["gmax"])
        tau_d.append(p["tau_d"])
        tau_r.append(p["tau_r"])
        tau_f.append(p["tau_f"])
        Uinc.append(p["Uinc"])
        pconn.append(p["pconn"])

params_synapses['Erev'] = np.asarray(Erev)
params_synapses['Gmax'] = 1.0 + np.zeros_like(Erev) # np.asarray(Gmax)
params_synapses['tau_d'] = np.asarray(tau_d)
params_synapses['tau_r'] = np.asarray(tau_r)
params_synapses['tau_f'] = np.asarray(tau_f)
params_synapses['Uinc'] = np.asarray(Uinc)
params_synapses['pconn'] = np.asarray(pconn)

target_params = pr.default_param4optimization

Duration = 10000
dt = 0.1
Cm = 3.0
t = np.arange(0, 10000, 0.1)
#t = t[100:]
tc = 0.5*Duration
Etarget = get_target_Esyn(t, tc, dt, theta_freq, animal_velocity, target_params)
print('params_synapses_["Erev"] 1 ', params_synapses["Erev"])


# plt.plot(t, Etarget)
#@tf.function
def Loss(X : tf.Variable, Duration, dt, Cm, animal_velocity, params_generators, params_synapses, target_params):
    global COUNTER
    print('params_synapses_["Erev"] 3 ', params_synapses["Erev"])



    theta_freq = params_generators['params'][0]['freq']
    Erev_hist = simulate(X, Duration, dt, Cm, animal_velocity, params_generators, params_synapses)

    tc = 0.5*Duration
    t = np.linspace(0, Duration, 100000)
    Etar = get_target_Esyn(t, tc, dt, theta_freq, animal_velocity, target_params)#tf.convert_to_tensor()
    #print(Etar)

    Etar = Etar[100:]
    Erev_hist = Erev_hist[100:]


    #L = 0.0
    L = tf.reduce_mean(tf.square(Etar - Erev_hist))
    #print(L)
    #L += np.sqrt( np.mean( (Etar - Erev_hist)**2) )
    #L = np.square(np.subtract(Etar,Erev_hist)).mean()
    return L

X1 = tf.Variable([74.37947807809046, 4.439029377217674, 1.052922557227097, 4.312755919523575, 7.619308476441775, 4.956026575387283, 2.334067214495775, -4.013870382013266, 7.430507250153729, 3.9270521343068197, -2.54624483101256, 3.741652532759068, 27.724704629937403, 5.4673925171809366, 6.0363955180268025, 85.90955322399513, 5.741508949738195, 1.0463906605011868, 81.01956945724712, 1.4737577916995033, 10.607406548313959, 4.038056986046165, 4.446894519893103, 14.89411683439329, 2.1582071283384465, -2.974530200852292, 9.732863583801855, 272.6876266344994, 467.5987551556552, 114.16680023266062, 243.1948186793486, 415.55488895442255, 263.91825685967996, 425.66093203559257, 1.7771137973655868, 359.89706346642055, 276.03857118708595, 0.4116616855840556])

#Erev_hist = np.array(simulate(X1, Duration, dt, Cm, animal_velocity, params_generators, params_synapses))
# plt.plot(t, Erev)
# plt.show()

#Loss(X1, Duration, dt, Cm, animal_velocity, params_generators, params_synapses, target_params)

optimizer = tf.optimizers.Adam(learning_rate=0.02)

# Количество шагов оптимизации
n_steps = 1000

# for step in range(n_steps):
#     optimizer.minimize(Loss, [X1])


# Цикл оптимизации
for step in range(n_steps):
    with tf.GradientTape() as tape:
        tape.watch(X1)
        print('params_synapses_["Erev"] 4 ', params_synapses["Erev"])
        loss = Loss(X1, Duration, dt, Cm, animal_velocity, params_generators, params_synapses, target_params)
    gradients = tape.gradient(loss, [X1])
    optimizer.apply_gradients(zip(gradients, [X1]))

    
    print(f"Step {step}, Loss: {loss.numpy()}")
    if step % 5 == 0:
        print("Optimized parameters:", X1.numpy())

# Итоговые параметры



'''
'''
