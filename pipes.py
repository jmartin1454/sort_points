class pipe:
    def __init__(self,rpipe,ypipe,zpipe,axisx):
        self.rpipe=rpipe
        self.ypipe=ypipe
        self.zpipe=zpipe
        self.axisx=axisx # 'x', 'y', or 'z'

class pipelist:
    def __init__(self):
        self.pipes=[]
        self.npipes=len(self.pipes)

    def add_pipe(self,rpipe,ypipe,zpipe,axisx):
        newpipe=pipe(rpipe,ypipe,zpipe,axisx)
        self.pipes.append(newpipe)
        self.npipes=len(self.pipes)
