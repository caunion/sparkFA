from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
from allensdk.brain_observatory.static_gratings import StaticGratings
from datetime import datetime

# 1. Setting up the cache information
print str(datetime.now())
print '------  1  ------'
boc = BrainObservatoryCache(cache=True, manifest_file='boc/manifest.json')
print 'Cache set up.'

# 2. Getting a list of target cells
print str(datetime.now())
print '------  2  ------'
cells = boc.get_cell_specimens(experiment_container_ids=[511500480])

# 3. Getting NWB files for the target exp_con_ids
print str(datetime.now())
print '------  3  ------'
exp = boc.get_ophys_experiments(experiment_container_ids=[511500480], stimuli=[stim_info.STATIC_GRATINGS])[0]
print str(datetime.now())
print '------  3A  ------'
data_set = boc.get_ophys_experiment_data(exp['id']) # Downloading step

# 4. Getting the sweep responses from the data set
print str(datetime.now())
print '------  4  ------'
# Warning will raise. Please ignore.
sg = StaticGratings(data_set)
sg.sweep_response.to_csv('sweep response.csv')
print str(datetime.now())
print '------  Finished  ------'