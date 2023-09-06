from gym.envs.registration import register
register(
     id='VoronoiWorld-v1',
     entry_point='InsectGym.Voronoi.VoronoiWorld:VoronoiWorld',
     max_episode_steps=10000
 )
register(
     id='VoronoiWorldGoal-v1',
     entry_point='InsectGym.Voronoi.VoronoiWorldGoal:VoronoiWorldGoal',
     max_episode_steps=10000
 )


register(
     id='ChopperScape-v1',
     entry_point='InsectGym.ChopperScape.ChopperScape:ChopperScape',
     max_episode_steps=10000
 )

register(
     id='MultiPassengerTaxi-v1',
     entry_point='InsectGym.MultiPassengerTaxi.MultiPassengerTaxi:MultiPassengerTaxiEnv',
     max_episode_steps=10000
 )

register(
     id='MultiPassengerTaxiPick-v1',
     entry_point='InsectGym.MultiPassengerTaxi.MultiPassengerTaxiPick:MultiPassengerTaxiPickEnv',
     max_episode_steps=10000
 )

register(
     id='MaggotInPetriDish-v1',
     entry_point='InsectGym.MaggotInPetriDish.MaggotInPetriDish:MaggotInPetriDishEnv',
     max_episode_steps=10000
 )