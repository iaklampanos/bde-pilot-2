from Dataset_transformations import Dataset_transformations
from scipy.spatial.distance import cosine
from sklearn.preprocessing import maxabs_scale

def descriptor_distance(descr,current_weather,euclidean=True):
    global_dist = []
    for d in xrange(len(descr)):
        descriptor = descr[d,:]
        local_dists = []
        for i in xrange(len(current_weather)):
            for lvl in xrange(3):
                if euclidean:
                    local_dists.append(np.linalg.norm(current_weather[i,lvl,:]-descriptor[i,lvl,:]))
                else:
                    local_dists.append(cosine(current_weather[i,lvl,:].flatten(),descriptor[i,lvl,:].flatten()))
        local_dists = np.array(local_dists)
        local_dists = np.mean(local_dists)
        global_dist.append((d,local_dists))
    global_dist = sorted(global_dist, key=lambda x: x[1], reverse=False)
    print global_dist
    return global_dist

def mean(v):
    return np.mean(v,axis=0)

def avg_descriptor_distance(descr,current_weather,euclidean=True):
    current_frames = []
    for i in xrange(len(current_weather)):
        frame = current_weather[i,:]
        frame = frame.reshape(1,1,1,3,64,64)
        ds = Dataset_transformations(frame,1000,frame.shape)
        ds.twod_transformation()
        ds.normalize()
        current_frames.append(m.get_hidden(ds._items.T))
    current_frames = np.array(current_frames).reshape(12,30)
    ddists = []
    for pos,d in enumerate(descr):
        desc_frames = []
        for i in xrange(len(current_weather)):
            frame = d[i,:]
            frame = frame.reshape(1,1,1,3,64,64)
            ds = Dataset_transformations(frame,1000,frame.shape)
            ds.twod_transformation()
            ds.normalize()
            desc_frames.append(m.get_hidden(ds._items.T))
        dists = []
        for i in xrange(len(current_frames)):
            if euclidean:
                dists.append(np.linalg.norm(current_frames[i]-desc_frames[i]))
            else:
                dists.append(cosine(current_frames[i].flatten(),desc_frames[i].flatten()))
        dists = np.array(dists)
        ddists.append((pos,np.mean(dists)))
    ddists = sorted(ddists, key=lambda x: x[1], reverse=False)
    print ddists
    return ddists


def kmeans_fit(desc,current_weather,euclidean=True):
    avg_l = np.asarray([mean(current_weather[:,0,:]),mean(current_weather[:,1,:]),mean(current_weather[:,2,:])])
    avg_l = avg_l.reshape(1,1,1,3,64,64)
    ds = Dataset_transformations(avg_l,1000,avg_l.shape)
    ds.twod_transformation()
    ds.normalize()
    z_current = m.get_hidden(ds._items.T)
    if euclidean:
        print c._link.predict(z_current)
        return c._link.predict(z_current)
    else:
        dists = []
        for i in xrange(c._n_clusters):
            dists.append((i, cosine(z_current,c._centroids[i])))
        dists = sorted(dists, key=lambda x: x[1], reverse=False)
        print dists
        return dists

def disp_dist(dist,disperions):
    readings = np.load('readings.npy')
    di_dist = []
    for pos,d in enumerate(disperions):
        disp = maxabs_scale(d.flatten(), axis=1)
        disp = disp.reshape(readings.shape)
        di_dist.append(1 - cosine(disp.flatten(),readings.flatten()))
    res = []
    for pos,d in enumerate(dist):
        print pos,di_dist[pos]
        print pos,d[1]
        res.append((d[0],d[1]*di_dist[pos]))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    print res
    return res

# Create hidden files for avg ae metric

# descr_path = '/mnt/disk1/thanasis/data/descriptors'
# list_descr = sorted(os.listdir(descr_path))
# for d in list_descr:
#     if d.endswith('.npy'):
#         desc_frames = []
#         descriptor = np.load(descr_path + '/' + d)
#         for i in xrange(len(current_weather)):
#             frame = descriptor[i,:]
#             frame = frame.reshape(1,1,1,3,64,64)
#             ds = Dataset_transformations(frame,1000,frame.shape)
#             ds.twod_transformation()
#             ds.normalize()
#             desc_frames.append(m.get_hidden(ds._items.T))
#         desc_frames = np.array(desc_frames)
#         desc_frames = desc_frames.reshape(13,30)
#         print desc_frames.shape
#         np.save(d.split('-GHT.npy')[0]+'-HIDDEN.npy',desc_frames)
#
#
# weather_path = '/mnt/disk1/thanasis/data/wrf/nc/npy'
# for current_date in sorted(os.listdir(weather_path)):
#   current_weather = np.load(weather_path + '/' + current_date)
#   levels = current_weather.shape[1] / 4096
#   current_frames = []
#   for i in xrange(len(current_weather)):
#       frame = current_weather[i,:]
#       frame = frame.reshape(1,1,1,levels,64,64)
#       ds = Dataset_transformations(frame,1000,frame.shape)
#       ds.twod_transformation()
#       ds.normalize()
#       current_frames.append(m.get_hidden(ds._items.T))
#   current_frames = np.array(current_frames).reshape(13,30)
#   print current_frames.shape
#   np.save(current_date+'-HIDDEN.npy',current_frames)
