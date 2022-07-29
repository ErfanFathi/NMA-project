from networks import (make_networks_d4pg,
                      make_networks_ddpg,
                      make_networks_dmpo)
from acme.agents.tf.d4pg import D4PG
from acme.agents.tf.ddpg import DDPG
from acme.agents.tf.dmpo import DistributionalMPO
from acme.tf import utils as tf2_utils
import sonnet as snt



def create_agent(env_spec, action_spec, lr, learner_logger, checkpoint, agent_name):
  
  # Note: optimizers can be passed only to the D4PG and DMPO agents.
  # The optimizer for DDPG is hard-coded in the agent class.
  policy_optimizer = snt.optimizers.Adam(lr)
  critic_optimizer = snt.optimizers.Adam(lr)
  
  # Create networks.
  if agent_name == "D4PG":
    policy_network, critic_network = make_networks_d4pg(action_spec)
    agent = D4PG(environment_spec=env_spec,
                policy_network=policy_network,
                critic_network=critic_network,
                observation_network=tf2_utils.batch_concat, # Identity Op.
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                logger=learner_logger,
                checkpoint=checkpoint)
  elif agent_name == "DMPO":
    policy_network, critic_network = make_networks_dmpo(action_spec)
    agent = DistributionalMPO(environment_spec=env_spec,
                              policy_network=policy_network,
                              critic_network=critic_network,
                              observation_network=tf2_utils.batch_concat,
                              policy_optimizer=policy_optimizer,
                              critic_optimizer=critic_optimizer,
                              logger=learner_logger,
                              checkpoint=False)
  elif agent_name == "DDPG":
    policy_network, critic_network = make_networks_ddpg(action_spec)
    agent = DDPG(environment_spec=env_spec,
             policy_network=policy_network,
             critic_network=critic_network,
             observation_network= tf2_utils.batch_concat, # Identity Op.
             logger=learner_logger,
             checkpoint=checkpoint)
  else:
    raise ValueError("{} Agent Not Found".format(agent_name))

  return agent