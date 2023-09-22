from Train import Train
from settings import Settings

if __name__ == "__main__":
    trainer = Train()
    trainer.train(Settings.data, 200)
    trainer.save("..\weights\weights.txt")