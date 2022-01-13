class MultiFields:
    def __init__(self, fields):
        """
        fields is a list of structured fields
        """
        self.__fields = fields
        
    @property
    def fields(self):
        return self.__fields
      
    def __getitem__(self, index):
        return self.__fields[index]  
    
    def __call__(self, points_list):
        return [f(p) for f, p in zip(self.__fields, points_list)]

        