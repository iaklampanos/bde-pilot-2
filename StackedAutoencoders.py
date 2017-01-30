
class StackedAutoencoders(object):

      def __init__(self,Autoencs,Hidden_layers_size):
          self.autoencs = Autoencs
          self.hidden_size = Hidden_layers_size
          self.hl = len(self.hidden_size)

      def train(self):
          for i in range(0,self.hl):
              self.autoencs[i].train()

      def get_train_encode(self,pos):
          for i in range(0,self.hl):
              if i == pos:
                 return self.autoencs[i].hidden

      def get_train_decode(self,pos):
          for i in range(0,self.hl):
              if i == pos:
                 return self.autoencs[i].decode

      def test(self,data):
          for i in range(0,self.hl):
              if i == 0:
                  self.autoencs[i].hidden = self.autoencs[i].get_hidden(data)
              else:
                  self.autoencs[i].hidden = self.autoencs[i].get_hidden(self.autoencs[i-1].hidden)
          for i in range(0,self.hl):
              if i == 0:
                  self.autoencs[i].decode = self.autoencs[i].get_output(data)
              else:
                  self.autoencs[i].decode = self.autoencs[i].get_output(self.autoencs[i-1].decode)
