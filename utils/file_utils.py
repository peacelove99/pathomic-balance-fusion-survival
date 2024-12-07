import pickle


def save_pkl(filename, save_object):
    writer = open(filename, 'wb')
    pickle.dump(save_object, writer)
    writer.close()
