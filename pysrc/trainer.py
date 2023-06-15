from Train import Train

if __name__ == "__main__":
    trainer = Train()
    trainer.train("..\..\data\data.csv", 200)
    trainer.save("..\..\weights\weights.txt")