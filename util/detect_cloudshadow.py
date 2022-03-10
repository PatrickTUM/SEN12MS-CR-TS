import numpy as np
import scipy
import scipy.signal as scisig


def rescale(data, limits):
    return (data - limits[0]) / (limits[1] - limits[0])


def normalized_difference(channel1, channel2):
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = 0.001  # checking for 0 divisions
    return subchan / sumchan


def get_shadow_mask(data_image):
    data_image = data_image / 10000.

    (ch, r, c) = data_image.shape
    shadowmask = np.zeros((r, c)).astype('float32')

    BB     = data_image[1]
    BNIR   = data_image[7]
    BSWIR1 = data_image[11]

    CSI = (BNIR + BSWIR1) / 2.

    t3 = 3/4 # cloud-score index threshold
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI))

    t4 = 5 / 6  # water-body index threshold
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB))

    shadow_tf = np.logical_and(CSI < T3, BB < T4)

    shadowmask[shadow_tf] = -1
    shadowmask = scisig.medfilt2d(shadowmask, 5)

    return shadowmask


def get_cloud_mask(data_image, cloud_threshold, binarize=False, use_moist_check=False):
    '''Adapted from https://github.com/samsammurphy/cloud-masking-sentinel2/blob/master/cloud-masking-sentinel2.ipynb'''

    data_image = data_image / 10000.
    (ch, r, c) = data_image.shape

    # Cloud until proven otherwise
    score = np.ones((r, c)).astype('float32')
    # Clouds are reasonably bright in the blue and aerosol/cirrus bands.
    score = np.minimum(score, rescale(data_image[1], [0.1, 0.5]))
    score = np.minimum(score, rescale(data_image[0], [0.1, 0.3]))
    score = np.minimum(score, rescale((data_image[0] + data_image[10]), [0.4, 0.9]))
    score = np.minimum(score, rescale((data_image[3] + data_image[2] + data_image[1]), [0.2, 0.8]))

    if use_moist_check:
        # Clouds are moist
        ndmi = normalized_difference(data_image[7], data_image[11])
        score = np.minimum(score, rescale(ndmi, [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = normalized_difference(data_image[2], data_image[11])
    score = np.minimum(score, rescale(ndsi, [0.8, 0.6]))

    boxsize = 7
    box = np.ones((boxsize, boxsize)) / (boxsize ** 2)

    score = scipy.ndimage.morphology.grey_closing(score, size=(5, 5))
    score = scisig.convolve2d(score, box, mode='same')

    score = np.clip(score, 0.00001, 1.0)

    if binarize:
        score[score >= cloud_threshold] = 1
        score[score < cloud_threshold]  = 0

    return score

# IN: [13 x H x W] S2 image (of arbitrary resolution H,W), scalar cloud detection threshold
# OUT: cloud & shadow segmentation mask (of same resolution)
# the multispectral S2 images are expected to have their default ranges and not be value-standardized yet
# cloud_threshold: the higher the more conservative the masks (i.e. less pixels labeled clouds/shadows)
def get_cloud_cloudshadow_mask(data_image, cloud_threshold):
    cloud_mask = get_cloud_mask(data_image, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(data_image)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0]  = 1

    return cloud_cloudshadow_mask
