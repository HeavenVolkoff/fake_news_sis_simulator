def edo_sis_k1(
    population,
    t0_infectious,
    transmission_rate, 
    recovery_rate,
    total_time,
):
  """
  Args:
      population: População inicial
      t0_infectious: Taxa inicial de infectados
      transmission_rate:
      recovery_rate:
      total_time:
  """
  # Proporção da população infectada
  t0_susceptibles = 1.0 - t0_infectious 
  INPUT = (t0_susceptibles, t0_infectious)

  def diff_eqs(INP,t):  
      Y=np.zeros((2))
      susceptibles = INP[0]
      infectious = INP[1]
      Y[0] = - transmission_rate * susceptibles * infectious + recovery_rate * infectious
      Y[1] = transmission_rate * susceptibles * infectious - recovery_rate * infectious
      return Y   # For odeint

  t_start = 0.0
  t_end = total_time  # Apenas mudei de 1000 para 100
  t_inc = 1
  t_range = np.arange(t_start, t_end+t_inc, t_inc)
  return population * spi.odeint(diff_eqs,INPUT,t_range)


def edo_sis_k2(
    population,
    t0_i1=0.2,
    t0_i2=0.0,
    tr_1=0.1, 
    tr_2=1.0, 
    rr=0.05, 
    t_end=200,
):
    """
    Args:
        population: População inicial
        t0_i1:  Porcentagem inicial de infectados em uma posição.
        t0_i2:  Porcentagem inicial de infectados em duas posições.
        tr_1: Taxa de infecção de casos com uma posição da timeline infectada
        tr_2: Taxa de infecção de casos com duas posições da timeline infectadas
        rr: Taxa de recuperação de casos com timeline infectada
    """
    S = 1.0 - t0_i1 - t0_i2 # Proporção da população inicial suscetível

    INPUT = (S, t0_i1, t0_i2)

    def diff_eqs(INP,t):  
      Y=np.zeros((3))
      V = INP
      susceptible = rr*V[1] - (tr_1*V[1] + tr_2*V[2]) * V[0]
      one_post_infected = (tr_1*V[1] + tr_2*V[2])*V[0] + rr*V[2] - (rr*V[1] + (tr_1*V[1]+tr_2*V[2])*V[1])
      two_post_infected = (tr_1*V[1] + tr_2*V[2])*V[1] - rr*V[2] 
      Y[0] = susceptible
      Y[1] = one_post_infected
      Y[2] = two_post_infected
      return Y   # For odeint

    t_start = 0.0; t_inc = 1.0
    t_range = np.arange(t_start, t_end+t_inc, t_inc)
    return population * spi.odeint(diff_eqs,INPUT,t_range)
 

def build_plot_k1(data):
    pl.plot(data[:,0], '-g', label='Susceptibles')
    pl.title('EDO - SIS - K=1')
    pl.xlabel('Time')
    pl.plot(data[:,1], '-r', label='Infectious')
    # pl.show()

    # Último valor RES[:, 0]
    print('Suscetíveis ao final:', data[-1, 0])

    # Último valor RES[:, 1]
    print('Infectados ao final:', data[-1, 1])
    

def build_plot_k2(data):
    pl.plot(data[:,1], 'orangered', label='Infected-1', linestyle='dashed')
    pl.plot(data[:,2], 'darkred', label='Infected-2', linestyle='dashed')
    pl.plot(data[:,0], '-g', label='Susceptible')
    pl.plot([x + y for x, y in zip(RES[:,1], RES[:,2])], '-r', label='Infectious')
    pl.legend()

    pl.xlabel('Time')
    pl.ylabel('Population %')
    # pl.show()
    
    print('Suscetíveis ao final:', data[-1, 0])
    print('Infectados com um post ao final:', data[-1, 1])
    print('Infectados com dois posts ao final:', data[-1, 2])
 
