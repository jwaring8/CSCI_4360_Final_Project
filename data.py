import pandas as pd

file_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
comments_df = pd.DataFrame()

for f in file_numbers:
    print("Reading JSON file " + str(f))
    path = 'data/SARC/comments_' + str(f) +'.json'
    temp_df = pd.read_json(path, orient='index')
    comments_df = comments_df.append(temp_df[['subreddit', 'text']])

    trainARR = pd.read_csv('data/SARC/train-balanced.csv').values
testARR = pd.read_csv('data/SARC/test-balanced.csv').values

for string in trainARR:
    string[0] = string[0].split('|')
for i in range(trainARR.shape[0]):
    trainARR[i,0][0] = trainARR[i,0][0].split()
    trainARR[i,0][1] = trainARR[i,0][1].split()
    trainARR[i,0][2] = trainARR[i,0][2].split()
    
for string in testARR:
    string[0] = string[0].split('|')
for i in range(testARR.shape[0]):
    testARR[i,0][0] = testARR[i,0][0].split()
    testARR[i,0][1] = testARR[i,0][1].split()
    testARR[i,0][2] = testARR[i,0][2].split()

data = {'Quote Text': [],
        'Response Text': [],
        'Subreddit': [],
        'Label': []}

for i in range(trainARR.shape[0]):
    for j in range(2):
        data['Quote Text'].append(comments_df.loc[[trainARR[i,0][0][0]]]['text'][0])
        data['Response Text'].append(comments_df.loc[[trainARR[i,0][1][j]]]['text'][0])
        data['Subreddit'].append(comments_df.loc[[trainARR[i,0][1][j]]]['subreddit'][0])
        data['Label'].append(trainARR[i,0][2][j])

SARC_DF = pd.DataFrame(data, columns=['Quote Text', 'Response Text', 'Subreddit', 'Label'])
SARC_DF.to_csv('data/SARC/SARC-train.csv', mode='a', index= False, encoding='utf-8', header = False)

data = {'Quote Text': [],
        'Response Text': [],
        'Subreddit': [],
        'Label': []}
        
for i in range(testARR.shape[0]):
    for j in range(2):
        data['Quote Text'].append(comments_df.loc[[testARR[i,0][0][0]]]['text'][0])
        data['Response Text'].append(comments_df.loc[[testARR[i,0][1][j]]]['text'][0])
        data['Subreddit'].append(comments_df.loc[[testARR[i,0][1][j]]]['subreddit'][0])
        data['Label'].append(testARR[i,0][2][j])


SARC_DF = pd.DataFrame(data, columns=['Quote Text', 'Response Text', 'Subreddit', 'Label'])
SARC_DF.to_csv('data/SARC/SARC-test.csv', mode='a', index= False, encoding='utf-8', header = False)