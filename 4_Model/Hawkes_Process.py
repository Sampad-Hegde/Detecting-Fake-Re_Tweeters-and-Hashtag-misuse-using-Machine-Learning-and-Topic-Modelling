from filters import get_retweet_df, get_timestamp, get_number_of_users, dateTimeCreator, generate_topic_vector, dateTimeCreator_new
from tick.hawkes import HawkesExpKern, HawkesSumExpKern, HawkesADM4, HawkesSumGaussians
import numpy as np
import warnings

warnings.simplefilter(action='ignore')


def get_hawkes_model(timestamps):
    decays = 0.3
    gofit = 'likelihood'
    penalty = 'l2'
    solver = 'agd'
    step = None
    tol = 1e-05
    max_iter = 1000
    verbose = False

    a_kernel = HawkesExpKern(decays, gofit=gofit, penalty=penalty, solver=solver, step=step, tol=tol, max_iter=max_iter,
                             verbose=verbose)
    a_kernel.fit(timestamps)

    baseline = a_kernel.baseline[0]
    adj = a_kernel.adjacency[0][0]

    return baseline, adj


def get_hawkes_model_2(timestamps):
    decays = 0.3
    penalty = 'l2'
    solver = 'agd'
    n_baselines = 1
    step = None
    period_length = None
    tol = 1e-05
    max_iter = 1000
    verbose = False

    a_kernel = HawkesSumExpKern(decays, penalty=penalty, n_baselines=n_baselines, period_length=period_length,
                                solver=solver, step=step, tol=tol, max_iter=max_iter, verbose=verbose)
    a_kernel.fit(timestamps)

    baseline = a_kernel.baseline[0]
    adj = a_kernel.adjacency[0][0]

    return baseline, adj


# HawkesADM4
# need 2-d array
def get_hawkes_model_3(timestamps):
    decays = np.array([0.3])
    lasso_nuclear_ratio = 0.5
    max_iter = 50
    tol = 1e-05
    n_threads = 1
    verbose = False
    rho = 0.1
    em_max_iter = 30
    em_tol = None

    a_kernel = HawkesADM4(decays, lasso_nuclear_ratio=lasso_nuclear_ratio, max_iter=max_iter, tol=tol,
                          n_threads=n_threads, verbose=verbose, rho=rho, em_max_iter=em_max_iter, em_tol=em_tol)
    a_kernel.fit(timestamps)
    baseline = a_kernel.baseline[0]
    adj = a_kernel.adjacency[0][0]
    return baseline, adj


# HawkesSumGaussians
# need 2-d array
def get_hawkes_model_4(timestamps):
    max_mean_gaussian = 5
    n_gaussians = 5
    step_size = 1e-07
    lasso_grouplasso_ratio = 0.5
    max_iter = 50
    tol = 1e-05
    n_threads = 1
    verbose = False
    em_max_iter = 30
    em_tol = None

    a_kernel = HawkesSumGaussians(max_mean_gaussian, n_gaussians=n_gaussians, step_size=step_size,
                                  lasso_grouplasso_ratio=lasso_grouplasso_ratio, max_iter=max_iter, tol=tol,
                                  n_threads=n_threads, verbose=verbose, em_max_iter=em_max_iter, em_tol=em_tol)
    a_kernel.fit(timestamps)
    baseline = a_kernel.baseline[0]
    adj = a_kernel.adjacency[0][0]

    return baseline, adj


def get_topic_vector(df, dict_genuine, dict_fake, lda_genuine, lda_fake, mode):
    num_topics = 10
    labels = []

    # LDA outputs
    user_topic_vectors = []
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Hawkes Process Started -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n\n')
    for i in range(get_number_of_users()):
        # from 0_Datset folder get each user tweets timeline and filter retweets from them
        i_user_df = get_retweet_df(i)
        if i_user_df.shape[0] < 2:
            continue

        # String Date Time to python Datetime library
        i_user_df.created_at = i_user_df.created_at.apply(dateTimeCreator)

        # time at which first retweet was made
        min_date = min(i_user_df['created_at'])

        # time to minnutes passed from the first retweet
        timestamps = i_user_df.created_at.apply(get_timestamp, origin_date=min_date).to_numpy()

        min_time = np.min(timestamps)
        max_time = np.max(timestamps)

        # Min-Max Scaler (0, 1)
        sorted_time = (np.sort(np.unique((timestamps - min_time) / max_time)))

        # fit the model and get the Hawkes Expression kernal model output
        BaseLine, adj_mat = get_hawkes_model(timestamps=[sorted_time])

        # append the results

        # baselines.append(BaseLine)
        if mode == 0:
            if df[df.user_id == i_user_df['user_id'][0]]['Annotation'].item() == 1 or \
                    df[df.user_id == i_user_df['user_id'][0]]['Tag'].item() == 1:
                labels.append(1)
            else:
                labels.append(0)
        elif mode == 1:
            labels.append(df[df.user_id == i_user_df['user_id'][0]]['Tag'].item())
        else:
            labels.append(df[df.user_id == i_user_df['user_id'][0]]['Annotation'].item())
        # hawkes_user_ids.append(i_user_df['user_id'][0])
        # adjs.append(adj_mat)

        # get topic vector for tweet in users.csv file
        tweet_text_list = df[df.user_id == i_user_df['user_id'][0]]['tweet_text'].item().split()
        topic_vector = generate_topic_vector(tweet_text_list,
                                             num_topics,
                                             dict_genuine,
                                             dict_fake,
                                             lda_genuine,
                                             lda_fake)
        topic_vector[-1] = BaseLine
        topic_vector[-2] = adj_mat

        user_topic_vectors.append(topic_vector)
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Hawkes Process Ended -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')

    return user_topic_vectors, labels


def Hawkes_main_driver():
    baselines = []
    adjs = []
    labels = []

    for i in range(get_number_of_users()):
        i_user_df = get_retweet_df(
            i)  # from 0_Datset folder get each user tweets timeline and filter retweets from them
        i_user_df.created_at = i_user_df.created_at.apply(
            dateTimeCreator)  # String Date Time to python Datetime library
        min_date = min(i_user_df['created_at'])  # time at which first retweet was made
        timestamps = i_user_df.created_at.apply(get_timestamp,
                                                origin_date=min_date).to_numpy()  # time to minnutes passed from the first retweet
        min_time = np.min(timestamps)
        max_time = np.max(timestamps)
        scaled_time = (np.sort(np.unique((timestamps - min_time) / max_time)))  # Min-Max Scaler (0, 1)
        BaseLine, adj_mat = get_hawkes_model(
            timestamps=[scaled_time])  # fit the model and get the Hawkes Expression kernal model output
        baselines.append(BaseLine)
        adjs.append(adj_mat)
        break
    return baselines, adjs, labels


def get_user_timeline_hawkes(df):
    df.created_at = df.created_at.apply(dateTimeCreator_new)  # String Date Time to python Datetime library
    min_date = min(df['created_at'])  # time at which first retweet was made
    timestamps = df.created_at.apply(get_timestamp,origin_date=min_date).to_numpy()  # time to minnutes passed from the first retweet
    min_time = np.min(timestamps)
    max_time = np.max(timestamps)
    scaled_time = (np.sort(np.unique((timestamps - min_time) / max_time)))  # Min-Max Scaler (0, 1)
    BaseLine, adj_mat = get_hawkes_model(timestamps=[scaled_time])  # fit the model and get the Hawkes Expression kernal model output
    return BaseLine, adj_mat

if __name__ == '__main__':
    Hawkes_main_driver()
