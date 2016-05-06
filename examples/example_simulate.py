import simulate
from scipy import misc
from PIL import Image
import numpy as np
from sklearn.mixture import GMM, VBGMM, DPGMM

if __name__ == "__main__":
    for func in [GMM, VBGMM, DPGMM]:
        print("Loading image...")
        #test_img = misc.face().astype(np.float32)
        test_img = np.array(Image.open("highschool2.png"))
        print("Getting pmfs...")
        pmfs = simulate.get_pmfs(test_img, func=func, n_components=6)
        print("Getting probability classes...")
        simulate.plot_prob_classes(pmfs, output_dir=func.__name__)
        print("Getting hard classes...")
        simulate.plot_hard_classes(pmfs, output_dir=func.__name__)
