import torch
import torch.nnfunctional as F
from envs import create_atari_env
from mod3 import ActorCritic
from torch.autogra import Variable

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param.grad = param.grad

def train(rank, params, shared_model,  optimizer):
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name)
    env.seed(params.seed + rank)
    model = ActorCritic(env.obeservation_space.shape[0], env.action_space)
    state = env.reset()
    state = torch.from_numoy(state)
    done = True
    episode_length = 0
    while True:

        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:

            cx = Variable(torch.zero(1, 256))
            hx = Variable(torch.zero(1, 256))
        else:

            cx =Variable(cx.data)
            hx = Variable(hx.data)
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):

            value, action_values, (hx, cx) = model((Variable(state.unzqueeze(0)), (hx, cx)))
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            state, reward, done = env.step(action.numpy())
            done = (done or episode_length >= params.max_epispde)
            reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                state = env.reset
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model(Variable(state.unzqueeze(0)), (hx, cx))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)#A(s,a) = Q(s,a)-V(s)
        for i in reversed(range(len(rewards))):
            R = params.gama * R + rewards[i]#cumulative_reward(R) = r0+gamma*r1+gama^2*r2+...+gamma^nb_steps*V(last_state)
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)#Q*(a*, s) = V*(s)
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD#gae = sum_i(gamma*tau)^i*TD(i)
            policy_loss = policy_loss - log_probs[i]*Variable(gae)-0.01*entropies[i]
        optimizer.zero_grad()
        (policy_loss+0.5 * value_loss).backward()
        torch.nn.utlis.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()