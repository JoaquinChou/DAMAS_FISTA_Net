import yaml

class Config():
    def __init__(self,configFile):
        self.configFile=configFile

        with open(self.configFile,'r') as f:
            self.config=yaml.load(f, Loader=yaml.FullLoader)
         
    
    def getConfig(self):
        if self.config:
            return self.config
        else:
            print("input error")


if __name__ == '__main__':
    con = Config("./config.yml").getConfig()
    print(con)
