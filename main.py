from train import train_montezuma, evaluate_model

if __name__ == "__main__":
    #train_montezuma()
    evaluate_model("models/montezuma_epsilonr_dqn_ep500.pth")