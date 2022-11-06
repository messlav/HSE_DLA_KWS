import logging
import multiprocessing as mp

import torch
import numpy as np
from torchaudio.io import StreamReader

logger = logging.getLogger(__file__)


def audio_stream(queue: mp.Queue):
    """
    Learn more about how to install and use streaming audio here
    https://pytorch.org/audio/stable/tutorials/streaming_api2_tutorial.html
    """

    streamer = StreamReader(src=":0", format="avfoundation")
    streamer.add_basic_audio_stream(frames_per_chunk=1000, sample_rate=16000)
    stream_iterator = streamer.stream(-1, 1)

    logger.info("Start audio streaming")
    while True:
        (chunk_,) = next(stream_iterator)
        logger.info("Put chunk to queue")
        queue.put(chunk_)


if __name__ == "__main__":
    model = torch.load("kws-10.pth", map_location='cpu').eval()

    ctx = mp.get_context("spawn")
    chunk_queue = ctx.Queue()
    streaming_process = ctx.Process(target=audio_stream, args=(chunk_queue,))

    streaming_process.start()
    results_buf = []
    while True:
        try:
            chunk = chunk_queue.get()
            chunk = chunk.view(1, -1)

            with torch.inference_mode():
                result = model(chunk)
            if result is not None:
                results_buf += [result > 0.9]

            if np.sum(results_buf[-50:]) >= 40:
                print("DETECTED KEY WORD")
                results_buf = []

        except KeyboardInterrupt:
            break
        except Exception as exc:
            raise exc

    streaming_process.join()
