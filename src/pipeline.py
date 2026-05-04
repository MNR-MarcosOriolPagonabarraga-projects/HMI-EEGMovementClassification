
class EEGPreprocessor:
    def __init__(self, 
                 apply_filter=True, l_freq=8.0, h_freq=30.0,
                 apply_notch=False, notch_freq=50.0,
                 apply_car=True,
                 apply_resample=False, resample_freq=250.0):
        
        self.apply_filter = apply_filter
        self.l_freq = l_freq
        self.h_freq = h_freq
        
        self.apply_notch = apply_notch
        self.notch_freq = notch_freq
        
        self.apply_car = apply_car
        
        self.apply_resample = apply_resample
        self.resample_freq = resample_freq

    def process(self, raw):
        # Always work on a copy to preserve the original data
        raw_proc = raw.copy()
        
        # 1. Notch Filter (Remove powerline noise)
        if self.apply_notch:
            raw_proc.notch_filter(freqs=self.notch_freq)
            
        # 2. Bandpass Filter (Causal)
        if self.apply_filter:
            raw_proc.filter(l_freq=self.l_freq, h_freq=self.h_freq, phase='minimum')
            
        # 3. Common Average Reference (Spatial Filter)
        if self.apply_car:
            raw_proc.set_eeg_reference('average')
            
        # 4. Downsampling (Do this LAST to prevent aliasing artifacts)
        if self.apply_resample:
            # MNE's resample automatically handles annotation timing adjustments
            raw_proc.resample(sfreq=self.resample_freq)
            
        return raw_proc