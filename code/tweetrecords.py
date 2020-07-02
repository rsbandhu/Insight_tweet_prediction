import parse_tweets
import os
import numpy as np
import time
import bz2
import pickle

import utils

class TweetRecords(object):

    def __init__(self, fname, batch_size, engagement_mode='retweet', embed_tokens_next=False):
        self.fname = fname
        self.get_file_handle()
        #self.file_handle = open(fname, 'r', encoding="utf-8")
        self.batch_size = batch_size
        self.max_batch_length = 125
        self.line = None
        self.parsed_fields = None
        self.embed_tokens_next = embed_tokens_next
        #Load all dicts for feature preprocessing
        (x1,x2) = self.load_all_dicts()
        
        self.language_to_num_dict = x1 #get the dict mapping language to a number 0, ... 61
        self.engaging_userid_hash_dict = x2 #dict mapping engaging userID to a hash mod 6000
        self.unique_id_tokenembedsdict = {} #dict that identies a tweet record uniquely and stores the tweet embeddings

        engaged_quantiles = utils.extract_follower_following_quantiles('../saved_features/engaged_usr_follower_following.pickle')
        self.engaged_usr_follower_quantiles = engaged_quantiles[:,0]
        self.engaged_usr_following_quantiles = engaged_quantiles[:,1]

        engaging_quantiles = utils.extract_follower_following_quantiles('../saved_features/engaging_usr_follower_following.pickle')
        self.engaging_usr_follower_quantiles = engaging_quantiles[:,0]
        self.engaging_usr_following_quantiles = engaging_quantiles[:,1]

        self.engagement_mode = engagement_mode

    def get_file_handle(self):
        self.file_handle = open(self.fname, 'r', encoding="utf-8")

    def load_all_dicts(self):
        print("loading lang dict")
        fhandle = bz2.BZ2File('../saved_features/language_to_number_dict_train_retweet_min_19.pickle', 'r')
        lang_dict = pickle.load(fhandle)
        fhandle.close()

        print("loading engaging_userid_hash_dict")
        fhandle = bz2.BZ2File('../saved_features/engaging_userid_hash_dict.pickle', 'r')
        engaging_userid_hash_dict = pickle.load(fhandle)
        fhandle.close()

        print("done loading all dict")
        return lang_dict, engaging_userid_hash_dict

    def tweet_records_generator(self, batch_size =1):
        self.line = self.file_handle.readline()
        try:
            if self.line:
                line = self.line
            yield line
        except StopIteration:
            pass

    def tokens_target_unpadded_gen(self):
        '''
        yield a batch of records record from the tweet file
        args:
        fname: path to the tweet file
        returns:
        a generator for the tweet records
        '''

        line = self.file_handle.readline()
        self.line = line
        try:
            while self.line:
                tokens_batch = []
                target_batch = []

                for _ in range(self.batch_size):
                    try:
                        if self.line:
                            token_list, yt = self.get_tokens_targets()
                            tokens_batch.append(token_list)
                            target_batch.append(yt)

                            self.line = self.file_handle.readline()
                    except StopIteration:
                        pass
                        
                yield tokens_batch, target_batch

        except StopIteration:
            pass

    def get_uniqueID(self):
        tweetID = self.get_tweet_id()
        #print(tweetID)
        engaged_userid = self.get_engaged_userid()
        engaging_userid = self.get_engaging_userid()

        unqiue_id = tweetID+'_'+engaged_userid+'_'+engaging_userid

        return unqiue_id

    def tokens_target_gen(self):
        '''
        yield a batch of tokens, targets and features from the tweet file
        args:
        fname: path to the tweet file
        returns:
        a generator for the tweet records
        '''

        line = self.file_handle.readline()
        self.line = line
        try:
            while self.line:
                tokens_batch = []
                target_batch = []
                features_batch = []

                for _ in range(self.batch_size):
                    try:
                        if self.line:
                            token_list, _ , yt = self.get_tokens_tweet_id_ytargets()
                            features = self.get_features_all_no_tokens_userid()

                            tokens_batch.append(token_list)
                            target_batch.append(yt)
                            features_batch.append(features)

                            self.line = self.file_handle.readline()
                    except StopIteration:
                        pass
                if self.embed_tokens_next:  #if tokens are going to be used for BERT
                    #pad tokens with 0 values to make each token list to have the same length        
                    if (len(target_batch) > 1):
                        tokens_batch = self.get_tokens_batch_padded(tokens_batch)
                    
                yield tokens_batch, target_batch, np.array(features_batch)

        except StopIteration:
            pass

    def tokenembeds_target_features_gen(self):
        '''
        yield a batch of token embeddings, targets and features from the tweet file
        args:
        fname: path to the tweet file
        returns:
        a generator for the tweet records
        '''

        line = self.file_handle.readline()
        self.line = line
        try:
            while self.line:
                tokens_embed_batch = []
                target_batch = []
                features_batch = []
                engaging_userid_hash_batch = []

                for _ in range(self.batch_size):
                    try:
                        if self.line:
                            self.parse_line()
                            uniqueID = self.get_uniqueID()
                            token_embedding = self.unique_id_tokenembedsdict[uniqueID]
                            y_targets = self.get_targets()
                            features = self.get_features_all_no_tokens_userid()
                            engaging_userid = self.get_engaging_userid()
                            engaging_userid_hash = self.engaging_userid_hash_dict[engaging_userid]
                            #engaging_userid_sparse = utils.create_onehot(6000, engaging_userid_hash)
                            if not (np.isnan(token_embedding).any()):
                                tokens_embed_batch.append(token_embedding)
                                target_batch.append(y_targets)
                                features_batch.append(features)
                                engaging_userid_hash_batch.append(engaging_userid_hash)

                            self.line = self.file_handle.readline()
                    except StopIteration:
                        pass    
                yield tokens_embed_batch, target_batch, np.array(features_batch), np.array(engaging_userid_hash_batch)

        except StopIteration:
            pass

    def tokens_tweetID_target_gen(self):
        '''
        yield a batch of records record from the tweet file
        args:
        fname: path to the tweet file
        returns:
        a generator for the tweet records
        '''

        line = self.file_handle.readline()
        self.line = line
        try:
            while self.line:
                tokens_batch = []
                tweet_id_batch = []
                target_batch = []
                follow_counts = []
                for _ in range(self.batch_size):
                    try:
                        if self.line:
                            token_list, tweet_id, yt = self.get_tokens_tweet_id_ytargets()
                            tokens_batch.append(token_list)
                            tweet_id_batch.append(tweet_id)
                            target_batch.append(yt)

                            self.line = self.file_handle.readline()
                    except StopIteration:
                        pass
                        
                if (len(tweet_id_batch) > 1):
                    tokens_batch = self.get_tokens_batch_padded(tokens_batch)
                #print(np.log(x1_max), np.log(x2_max), np.log(x3_max), np.log(x4_max))
                yield tokens_batch, tweet_id_batch, target_batch
        except StopIteration:
            pass

    def parse_line(self):
        line = self.line
        line.strip()
        field_data = line.split("\x01")
        self.parsed_fields = field_data

        return field_data

    def get_tokens(self, line):
        tokens = self.parse_line()[0]
        tokens_list = tokens.split()
        return tokens_list

    def get_tokens(self, token_field):
        text_tokens = token_field.split()
        if self.embed_tokens_next:
            tokens_list = [int( y ) for y in text_tokens] # return integer values
        else:
            tokens_list = [y for y in text_tokens] #return char values for tokens

        return tokens_list
    
    def get_tweet_id(self):
        #print(self.parsed_fields[2])
        return self.parsed_fields[2]

    def get_language(self):
        return self.parsed_fields[7]

    def get_tweet_time(self, idx):
        '''
        returns the (year, month, day, hr, minute, sec) from the unix time stamp
        :param parsed_fields:
        :return:
        '''
        unix_timestamp = self.parsed_fields[idx]
        if (len(unix_timestamp) > 0):
            (year, month, day, hr, minute, weekday) = utils.extract_year_month_day_hour_min_weekday(unix_timestamp)
            return (year, month, day, hr, minute, weekday)
        else:
            return None

    def get_targets(self):
        '''
        returns the target values for each engagement level for a tweet
        :param
        parsed_fields: original tweet parsed into different fields
        :return:
        y_targets: list of int, 0 or 1 representing target value for each engagement level
        '''
        y_targets = []
        parsed_fields = self.parsed_fields
        for i in range(4): # we have 4 targets, one for each engagement
            if len(parsed_fields[-4+i] ) > 0:
                y_target = 1
            else:
                y_target = 0
            y_targets.append(y_target)
        return y_targets

    def get_tokens_tweet_id_ytargets(self):
        '''
        get the tokens as a list and the tweet id
        :param
        line:
        :return:
        tokens_list: list of integers, each int representing a token
        tweet_id: id of the tweet
        y_targets: list of int, 0 or 1 representing target value for each engagement level
        '''

        parsed_fields = self.parse_line() #parse a tweet into different fields
        tokens_list = self.get_tokens(parsed_fields[0]) #
        tweet_id = self.get_tweet_id()
        y_targets = self.get_targets()

        return tokens_list, tweet_id, y_targets

    def get_engaged_userid(self):
        return self.parsed_fields[9]
    
    def get_engaging_userid(self):
        return self.parsed_fields[14]

    def get_tokens_targets(self):
        
        tweetrecords = self.parse_line()
        tokens = tweetrecords[0]
        tokens_list = tokens.split()
    
        y_targets = []
        #last 4 fields in the tweet indicate engagement time, if any
        #for each type, return 1 if there is an engagement time
        # else return 0x
        for i in range(1,5):
            if len(tweetrecords[-i]) > 0:
                y_target = 1
            else:
                y_target = 0
            y_targets.append(y_target)

        return tokens_list, y_targets

    def get_tokens_batch_padded(self, tokens_list):
        max_length = 0
        batch_size = len(tokens_list) # number of tweet records
        tokens_batch = []
        for i in range(batch_size):
            tokens_in_record_i = len(tokens_list[i])
            t_tokens = tokens_list[i]
            if tokens_in_record_i > max_length:
                max_length = tokens_in_record_i
            if (tokens_in_record_i > self.max_batch_length):
                t_tokens = tokens_list[i][:self.max_batch_length-1]+ [102]
                max_length = self.max_batch_length
            tokens_batch.append(t_tokens)

        for j in range( batch_size ):
            if len( tokens_batch[j] ) < max_length:
                tokens_batch[j] = tokens_batch[j][:-1] + [0] * (max_length - len( tokens_batch[j] )) + [102]

        return tokens_batch

    def follow_count_engageduser_10_11(self):
        return (int(self.parsed_fields[10]), int(self.parsed_fields[11]))

    def follow_count_engaginguser_15_16(self):
        return (int(self.parsed_fields[15]), int(self.parsed_fields[16]))

    def get_media_present_3(self):
        '''
        :return:integer = 1 or 0 corresponding to if media is present or not
        '''
        return (int(len(self.parsed_fields[3]) > 0))

    def get_link_present_4(self):
        '''
        :return:integer = 1 or 0 corresponding to if media is present or not
        '''
        return (int(len(self.parsed_fields[4]) > 0))

    def get_domain_present_5(self):
        '''
        :return:integer = 1 or 0 corresponding to if domain is present or not
        '''
        return (int(len(self.parsed_fields[5]) > 0))

    def engaged_user_verified_12(self):
        return int(self.parsed_fields[12] == 'true')

    def engaging_user_verified_17(self):
        return int(self.parsed_fields[17] == 'true')

    def engagee_follows_engager(self):
        return int( self.parsed_fields[19] == 'true' )

    def get_features_tweet(self):
        '''
        returns a list of features of length 34 
        Each item in the list is 0 or 1
        '''
        features = []
        
        features.append(self.get_media_present_3()) #len(feature) =1
        features.append(self.get_link_present_4())  #len(feature) =2
        features.append(self.get_domain_present_5()) #len(feature) =3

        language = self.get_language()
        lang_num = self.language_to_num_dict[language]
        lang_features = utils.create_onehot(62, lang_num)
        features += lang_features  #len(feature) =3 + 62 = 65

        (year, month, day, hr, minute, weekday) = self.get_tweet_time(8)
        
        weekday_features = utils.create_onehot(7, weekday)
        features += weekday_features #len(feature) =65 + 7 = 72
        hr_features = utils.create_onehot(24, hr)
        features += hr_features #len(feature) = 72 + 24 = 96
        
        return features

    def get_features_engaging_user(self):
        '''
        returns a list of features of length 77 
        Each item in the list is 0 or 1
        '''

        features = []
        features.append(self.engaging_user_verified_17()) #len(feature) =1
        (follower_count, following_count) = self.follow_count_engaginguser_15_16()

        bucket_count, follower_bucket = utils.create_hist_bucket_index(self.engaging_usr_follower_quantiles, follower_count)
        follower_onehot = utils.create_onehot(bucket_count, follower_bucket)
        features = features+ follower_onehot #len(feature) =1 +32 = 33
        
        bucket_count, following_bucket = utils.create_hist_bucket_index(self.engaging_usr_following_quantiles, following_count)
        following_onehot = utils.create_onehot(bucket_count, following_bucket)
        features = features + following_onehot #len(feature) =33 +32 = 65
    
        #add one hot features for the year that engaging user account was created
        (year, month, day, hr, minute, weekday) = self.get_tweet_time(18)
        year_buckets = [2010+ i for i in range(11)]
        bucket_count, year_bucket = utils.create_hist_bucket_index(year_buckets, year)
        year_features = utils.create_onehot(bucket_count, year_bucket)
        features = features + year_features #len(feature) = 65 +12 = 77

        return features

    def get_features_engagedwith_user(self):
        '''
        returns a list of features of length 77 
        Each item in the list is 0 or 1
        '''
        features = []
        features.append(self.engaged_user_verified_12()) #len(feature) =1

        
        (follower_count, following_count) = self.follow_count_engageduser_10_11()

        #add one hot feature for number of followers of engaged with user
        bucket_count, follower_bucket = utils.create_hist_bucket_index(self.engaged_usr_follower_quantiles, follower_count)
        follower_onehot = utils.create_onehot(bucket_count,follower_bucket)
        features = features+ follower_onehot #len(feature) =1 +32 = 33
        
        #add one hot feature for number of users that  engaged with user is following
        bucket_count, following_bucket = utils.create_hist_bucket_index(self.engaged_usr_following_quantiles, following_count)
        following_onehot = utils.create_onehot(bucket_count, following_bucket)
        features = features + following_onehot #len(feature) =33 +32 = 65

        #add one hot features for the year that engaged with user account was created
        (year, month, day, hr, minute, weekday) = self.get_tweet_time(13)
        year_buckets = [2010+ i for i in range(11)]
        bucket_count, year_bucket = utils.create_hist_bucket_index(year_buckets, year)
        year_features = utils.create_onehot(bucket_count, year_bucket)
        features = features + year_features #len(feature) = 65 +12 = 77

        return features

    def get_features_engagement(self, engage_idx):
        '''
        returns a list of features of length 58. 
        Each item in the list is 0 or 1
        '''

        features = []

        features.append(self.engagee_follows_engager()) #len(feature) =1
        
        #add one hot features for the year that engaged with user account was created
        engage_time= self.get_tweet_time(engage_idx)

        if engage_time is not None:
            (year, month, day, hr, minute, weekday)  = engage_time
            weekday_features = utils.create_onehot(7, weekday)
            features += weekday_features #len(reply_features) = 1 + 7 = 8
            hr_features = utils.create_onehot(24, hr)
            features += hr_features #len(feature) = 8 + 24 = 32

            tweet_engage_time_diff = (int(self.parsed_fields[engage_idx]) - int(self.parsed_fields[8]) ) // 3600 +1
            if  tweet_engage_time_diff > 24:
                tweet_engage_time_diff = 25
            tweet_engage_time_diff_features = utils.create_onehot(26, tweet_engage_time_diff)

            features += tweet_engage_time_diff_features #len(feature) = 32 + 26 = 58
            tweet_time = self.get_tweet_time(8)
            
        else:
            features += [0]*57

        return features

    def get_features_all_no_tokens_userid(self, ):
        '''
        returns all features for one of four engagement type
        Doesn't include tokens and user id features
        
        Returns:
        list of integers of length = 246
        '''
        
        engagement_mode = self.engagement_mode
        features_no_token_userid = []
        #add tweet features
        tweet_features = self.get_features_tweet()
        features_no_token_userid += tweet_features

        #add engaging user features
        engaging_user_features = self.get_features_engaging_user()
        features_no_token_userid += engaging_user_features

        #add engaged with user features
        engaged_with_usr_features = self.get_features_engagedwith_user()
        features_no_token_userid += engaged_with_usr_features

        #add engagement mode = reply features
        if engagement_mode == 'reply':
            reply_features = self.get_features_engagement(20)
            features_no_token_userid += reply_features 

        #add engagement mode = retweet features
        elif engagement_mode == 'retweet':
            retweet_features = self.get_features_engagement(21)
            features_no_token_userid += retweet_features 

        elif engagement_mode == 'like':
            retweetcomment_features = TR.get_features_engagement(22)
            features_no_token_userid += retweetcomment_features

        else:
            like_features  = TR.get_features_engagement(23)
            features_no_token_userid += like_features

        return features_no_token_userid
         
    def close_file(self):
        self.file_handle.close()

def main():
    file_path = 'test.txt'
    #train_file = "train_first_1000.tsv"
    #train_file = "val.tsv"
    #train_file = "training.tsv"
    train_file = "train_retweet_min_19.tsv"
    data_dir = "../data"
    file_path = os.path.join(data_dir, train_file)
    follow_counts_all = []
    t1 = time.time()
    A = TweetRecords(file_path,10)


    """ 
    ## Testing generator 'tokens_tweetID_target_gen'
    X_gen = A.tokens_tweetID_target_gen()
    X,y,z, follow_counts = next(X_gen)
    follow_counts_all.append( follow_counts )

    iter_count = 0
    while X:
        try:
            X,y,z,follow_counts = next(X_gen)
            if iter_count%10 == 0:
                print(iter_count, "  ::  ", len(y), follow_counts)
            #print(" **  ", len(X), len(y),"\n")
            follow_counts_all.append(follow_counts)
            iter_count += 1
        except StopIteration:
            break
    A.close_file() """

    ## Testing generator 'tokens_tweetID_target_gen'

    data_gen = A.tokenembeds_target_features_gen()
    token_ebmeds, y_targets, features = next(data_gen)

    iter_count = 0 
    while token_ebmeds:
        try:
            print(type(token_ebmeds), type(y_targets), type(features))
            token_ebmeds, y_targets, features = next(data_gen)

            iter_count += 1
            if (iter_count > 5):
                break
        except StopIteration:
            break

    t2 = time.time()

    print("total time taken to parse {} : {}".format(train_file, t2-t1))
    #np.savez_compressed( './saved_features/follow_count_all', follow_counts_all )
if __name__ == "__main__":
    main()
