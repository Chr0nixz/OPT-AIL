from agent.bc import BC
from agent.cmil import CMIL
from agent.dac import DAC
from agent.hyper import HYPER
from agent.mb_ail import MB_AIL
from agent.mbpo import MBPO
from agent.opt_ail import OPT_AIL
from agent.sac import SAC


def make_agent(env, args):
    obs_dim = env.observation_space.shape[0]

    if args.agent.name == "sac":
        print('--> Using SAC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = SAC(obs_dim, action_dim, action_range, args.train.batch, args)
    elif args.agent.name == "opt_ail":
        print('--> Using OPT-AIL agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = OPT_AIL(obs_dim, action_dim, action_range, args.train.batch, args)
    elif args.agent.name == "mb_ail":
        print('--> Using MB-AIL agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = MB_AIL(obs_dim, action_dim, action_range, args.train.batch, args)
    elif args.agent.name == "hyper":
        print('--> Using HyPER agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = HYPER(obs_dim, action_dim, action_range, args.train.batch, args)
    elif args.agent.name == "mbpo":
        print('--> Using MBPO agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = MBPO(obs_dim, action_dim, action_range, args.train.batch, args)
    elif args.agent.name == "bc":
        print('--> Using BC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = BC(obs_dim, action_dim, action_range, args.train.batch, args)
    elif args.agent.name == "cmil":
        print('--> Using CMIL agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = CMIL(obs_dim, action_dim, action_range, args.train.batch, args)
    else:
        print('--> Using DAC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = DAC(obs_dim, action_dim, action_range, args.train.batch, args)


    return agent
