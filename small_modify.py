import pandas as pd

"""
case_id,slide_id,label
67c271bc86db14ba,2cfef9a2d856f47e84c17b623b76,present
0702eb533c6dfcaa,435845a0ced30df3fb7b9eca1c40,not present
1b2b171a6c03ed08,d64b468114c8beb67c1aa0ac037e,not present
37b763e292205a43,b0bbb1d6c8790c6f38bdc6fd8e4b,not present
3f31387e4f55e4e3,e2de253b840917377c83c19121a8,not present
"""
# add batch1 _ to the slide_id
df = pd.read_csv("../../Data/proscia_metadata/Ruby_Robotics_SOW-001_Diff_Quik_Batch_1_KRAS_v2.csv")
df['slide_id'] = 'batch1_' + df['slide_id']
df.to_csv("../../Data/proscia_metadata/Ruby_Robotics_SOW-001_Diff_Quik_Batch_1_KRAS_v2.csv", index=False)