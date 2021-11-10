from gym.envs.registration import register
register(
     id='VoronoiWorld-v1',
     entry_point='InsectGym.Voronoi:VoronoiWorld',
     max_episode_steps=10000
 )
register(
     id='ChopperScape-v1',
     entry_point='InsectGym.ChopperScape:ChopperScape',
     max_episode_steps=10000
 )