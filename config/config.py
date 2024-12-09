import yaml

class Config:
    def __init__(self):
        with open("config.yaml", "r") as file:
            self.config = yaml.safe_load(file)

    def milvus(self, key):
        return self.config["milvus"][key]

    def embedding(self, key):
        return self.config["embedding"][key]

    def chat_model(self, key):
        return self.config["chat-model"][key]

    def loader(self, key):
        return self.config["loader"][key]
