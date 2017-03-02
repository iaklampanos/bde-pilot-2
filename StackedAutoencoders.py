
class StackedAutoencoders(object):

      def __init__(self,_autoencs,Hidden_layers_size):
          self._autoencs = _autoencs
          self._hidden_size = Hidden_layers_size
          self._hl = len(self._hidden_size)
          self._hidden_list = range(0,self._hl)

      def train(self):
          for i in range(0,self._hl):
              self._autoencs[i].train()

      def get_hidden(self,pos=None):
          if not(pos is None):
              i = self._hidden_list.index(pos)
              return self._autoencs[i].hidden
          else:
              return self._autoencs[self._hl-1].hidden

      def get_output(self,pos=None):
          if not(pos is None):
              i = self._hidden_list.index(pos)
              return self._autoencs[i].decoded
          else:
              return self._autoencs[self._hl-1].decoded

      def test(self,data):
          for i in self._hidden_list:
              if i == 0:
                  self._autoencs[i].hidden = self._autoencs[i].get_hidden(data)
              else:
                  self._autoencs[i].hidden = self._autoencs[i].get_hidden(self._autoencs[i-1].hidden)
          for i in self._hidden_list:
              if i == 0:
                  self._autoencs[i].decode = self._autoencs[i].get_output(data)
              else:
                  self._autoencs[i].decode = self._autoencs[i].get_output(self._autoencs[i-1].decode)
