import src.environment as environment

def test_preprocess():
    env = environment.MontezumaEnvironment(render_mode="rgb_array")
    env.reset()
    env.step(1)
    env.display_frame_stack()
    env.close()

if __name__ == "__main__":
    test_preprocess()