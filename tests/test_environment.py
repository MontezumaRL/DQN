import src.environment as environment
from gym.utils.play import play

def test_environment_play():
    env = environment.MontezumaEnvironment(render="rgb_array")
    env.reset()
    play(env, zoom=5, fps=15)
    env.close()

def test_environment_lives():
    env = environment.MontezumaEnvironment(render_mode="rgb_array")
    env.reset()
    #env.step(3)
    #env.step(3)
    #env.step(3)
    #env.step(3)
    #env.step(3)

    env.display_frame_stack()
    print(env.lives)
    print(env._get_agent_position(env.frame_stack))
    env.close()


if __name__ == "__main__":
    test_environment_lives()