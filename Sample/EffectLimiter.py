import numpy

class CreateLimiter:
    def __init__(self, threshold_in_db, attack_coeff=0.9, release_coeff=0.992, delay=10):
        self.delay_index = 0
        self.envelope = 0
        self.gain = 1
        self.delay = delay
        self.delay_line = numpy.zeros(delay)
        self.release_coeff = release_coeff
        self.attack_coeff = attack_coeff
        self.threshold = numpy.int16((10 ** (threshold_in_db/20))*32767)
        print(self.threshold)

    def apply(self, signal):
        for idx, sample in enumerate(signal):
            self.delay_line[self.delay_index] = sample
            self.delay_index = (self.delay_index + 1) % self.delay

            # calculate an envelope of the signal
            self.envelope  = max(abs(sample), self.envelope*self.release_coeff)

            if self.envelope > self.threshold:
                target_gain = self.threshold / self.envelope
            else:
                target_gain = 1.0

            # have self.gain go towards a desired limiter gain
            self.gain = ( self.gain*self.attack_coeff +
                          target_gain*(1-self.attack_coeff) )

            # limit the delayed signal
            signal[idx] = self.delay_line[self.delay_index] * self.gain
        return signal