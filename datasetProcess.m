%%
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadxdf('~\data\subject\s1_calibration.xdf' , 'streamtype', 'EEG', 'exclude_markerstreams', {});
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'gui','off'); 
EEG = eeg_checkset( EEG );
%%
EEG = pop_reref( EEG, [63 64] );
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','off','gui','off'); 
eeglab redraw;
%%
% extract epochs，then save data and events
EEG = pop_epoch( EEG, {'0' '1'}, [-1  4], 'newname', 'MI epochs', 'epochinfo', 'yes');
EEG = eeg_checkset( EEG );
type = {EEG.event.type}.';
N = length(type);
labels = [];
for i=1:N
    labels = [labels str2num(type{i})];
end
idxes = (labels == 0) + (labels == 1) + (labels == 2) + (labels == 3);
labels = labels(logical(idxes));
EEG_data = EEG.data;
save s1_calibration EEG_data labels