from matplotlib.pyplot import close


class PlotTestCase(object):

    def tearDown(self):
        close('all')
