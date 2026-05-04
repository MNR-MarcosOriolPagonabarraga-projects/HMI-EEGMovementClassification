

class EEGPreprocessor:
    def __init__(self, apply_filter=True, apply_car=True):
        self.apply_filter = apply_filter
        self.apply_car = apply_car

    def process(self, raw):
        # We always work on a copy so we don't destroy the original data
        raw_proc = raw.copy()
        
        if self.apply_filter:
            # Using the exact same parameters your dataset builder will use
            raw_proc.filter(l_freq=8.0, h_freq=30.0, phase='minimum')
            
        if self.apply_car:
            raw_proc.set_eeg_reference('average')
            
        return raw_proc