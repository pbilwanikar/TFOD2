import tensorflow as tf
import cv2
import os
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder


class Recommendation:

    def __init__(self, model, config, labelmap,category_index, checkpoint,detection_model,video):
        self.model = model
        self.config = config
        self.labelmap = labelmap
        self.checkpoint = checkpoint
        self.category_index = category_index
        self.detection_model = detection_model
        self.video = video

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    # def points(coords):
    #     for i in range(len(coords)):
    #         xmin, ymin, xmax, ymax = coords[i][1:]
    #
    #         pt1 = (xmin, ymin)
    #         pt2 = (xmin, ymax)
    #         pt3 = (xmax, ymin)
    #         pt4 = (xmax, ymax)
    #
    #         return (pt1, pt2, pt3, pt4)  # Eg --> ((2,2), (2, 4), (4, 2), (4, 4))


    def boundBox(self):
        cap = cv2.VideoCapture(self.video)
        # while True:
        #     ret, frame = cap.read()
        #     image_np = np.array(frame)
        while cap.isOpened():
            ret, image_np = cap.read()
            if not ret:
                break
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = self.detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            coords = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

            cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
            print(coords)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

def main():
    model = "exported-models/my_model/saved_model"
    config = "models/my_model_frr50/pipeline.config"
    labelmap = "annotations/label_map.pbtxt"
    checkpoint = "models/my_model_frr50"
    video = "Video/overpass.mp4"


    configs = config_util.get_configs_from_pipeline_file(config)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(checkpoint, 'ckpt-8')).expect_partial()
    category_index = label_map_util.create_category_index_from_labelmap(labelmap)

    recommend = Recommendation(model=model, config=config,
                               labelmap=labelmap,
                               checkpoint=checkpoint,
                               detection_model=detection_model,
                               category_index=category_index,
                               video=video)

    recommend.boundBox()



if __name__ == "__main__":
    main()




