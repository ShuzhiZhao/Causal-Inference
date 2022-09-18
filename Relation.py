import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def drawScatter(df,index):
    ERPs = ['N1_Mean', 'N1_std', 'N1_median', 'N1_Max', 'N1_Power', 'P2_Mean','P2_std', 'P2_median', 'P2_Max', 'P2_Power']
    SC_Names = ['SC_Mean', 'SC_std','SC_median', 'SC_Max', 'SC_Power']
    for ERP in ERPs:
        for SC_Name in SC_Names :
            df1 = df[[ERP,SC_Name]].astype(float)
            r1 = df1.corr().iat[1,0]
            # Plot
#             sns.set_style("white")
            gridobj1 = sns.lmplot(x=ERP, y=SC_Name, data=df1)
            # Decorations
            xmin = df1[ERP].min()-abs(df1[ERP].min()*0.06)
            xmax = df1[ERP].max()+abs(df1[ERP].min()*0.05)
            ymin = df1[SC_Name].min()-abs(df1[SC_Name].min()*0.5)
            ymax = df1[SC_Name].max()+abs(df1[SC_Name].min()*0.5)
            gridobj1.set(xlim=(xmin,xmax), ylim=(ymin,ymax))
            plt.xlabel(ERP,fontsize=9)
            plt.ylabel(SC_Name,fontsize=9)
            plt.title(index+" Pearson Correlation "+str(r1), fontsize=6)
            if abs(r1) > 0.4 :
                if 'S_temporal' in index or 'S_precentral' in index or 'G_front_inf' in index:
                    plt.savefig('/media/lhj/Momery/causalML/IFG_Source_Causal/data/Relation1/Pearson_'+index+ERP+SC_Name+'_corr.jpg',dpi=200,bbox_inches='tight')   
            plt.clf()
            plt.close()
    
# load data from N1P2.csc
data = pd.read_csv('/media/lhj/Momery/causalML/IFG_Source_Causal/data/N1P2.csv',dtype=object)
label = data.loc[:,['TimeSerials']].groupby(by='TimeSerials').count()
subName = data.loc[:,['Name']].groupby(by='Name').count()
N1 = data.loc[data['TimeSerials']=='N1[200:400]',['Mean','std','median','Max','Power']]
P2 = data.loc[data['TimeSerials']=='P2[400:600]',['Mean','std','median','Max','Power']]
for index in label.index:
    if 'N1[200:400]' in index and 'P2[400:600]' in index :
        print('skip ...')
    else: 
        SC = data.loc[data['TimeSerials']==index,['Mean','std','median','Max','Power']]        
    df = pd.DataFrame({'N1_Mean': N1['Mean'].values.tolist(),
                       'N1_std': N1['std'].values.tolist(),
                       'N1_median': N1['median'].values.tolist(),
                       'N1_Max': N1['Max'].values.tolist(),
                       'N1_Power': N1['Power'].values.tolist(),
                       'P2_Mean': P2['Mean'].values.tolist(),
                       'P2_std': P2['std'].values.tolist(),
                       'P2_median': P2['median'].values.tolist(),
                       'P2_Max': P2['Max'].values.tolist(),
                       'P2_Power': P2['Power'].values.tolist(),
                       'SC_Mean': SC['Mean'].values.tolist(),
                       'SC_std': SC['std'].values.tolist(),
                       'SC_median': SC['median'].values.tolist(),
                       'SC_Max': SC['Max'].values.tolist(),
                       'SC_Power': SC['Power'].values.tolist()
                      })
    print(df.columns,'\n\n')
    drawScatter(df,index)
    
    