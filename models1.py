from torch import nn
structuresize=1656
drugsize=881
DDINumber=1318

class CP(nn.Module):
    def __init__(self,R):
        super(CP, self).__init__()
        #A、C
        self.Structureembeding=nn.Linear(structuresize,R)
        self.interactionembeding=nn.Linear(DDINumber,R)
        #aggregate information of rank-one tensor
        self.getresult=nn.Linear(R,1)

    def forward(self,drug1,drug2,interaction):
        #drug1*A
        side_1 = self.Structureembeding(drug1)
        #drug2*A
        side_2=self.Structureembeding(drug2)
        #interaction*C
        interaction=self.interactionembeding(interaction)
        result=self.getresult(side_1*side_2*interaction)
        return result,interaction





