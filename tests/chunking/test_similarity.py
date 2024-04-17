from alinet.chunking.model import TimeChunk
from alinet.chunking.similarity import get_similarity_scores


class TestSimilarityScores:
    """Unit tests for the get_similarity_scores function"""

    error_range = 0.01

    def test_given_slide_end_time_within_transcript_time_then_similar(self):
        transcript_chunks = [TimeChunk("lorem ipsum", start_time=0.0, end_time=10.0)]
        slide_chunks = [
            TimeChunk("lorem ipsum", start_time=0.0, end_time=5.0),
        ]

        sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)
        diff = 1.0 - sim_scores[0]

        assert diff < self.error_range

    def test_given_slide_start_time_within_transcript_time_then_similar(self):
        transcript_chunks = [TimeChunk("lorem ipsum", start_time=0.0, end_time=10.0)]
        slide_chunks = [
            TimeChunk("lorem ipsum", start_time=5.0, end_time=10.0),
        ]

        sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)
        diff = 1.0 - sim_scores[0]

        assert diff < self.error_range

    def test_given_slide_time_equals_transcript_time_then_similar(self):
        transcript_chunks = [TimeChunk("lorem ipsum", start_time=0.0, end_time=10.0)]
        slide_chunks = [
            TimeChunk("lorem ipsum", start_time=0.0, end_time=10.0),
        ]

        sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)
        diff = 1.0 - sim_scores[0]

        assert diff < self.error_range

    def test_given_slide_times_equals_transcript_time_then_similar(self):
        transcript_chunks = [TimeChunk("lorem ipsum", start_time=0.0, end_time=10.0)]
        slide_chunks = [
            TimeChunk("lorem", start_time=0.0, end_time=5.0),
            TimeChunk("ipsum", start_time=5.0, end_time=10.0),
        ]

        sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)
        diff = 1.0 - sim_scores[0]

        assert diff < self.error_range

    def test_given_slide_times_within_transcript_time_then_similar(self):
        transcript_chunks = [
            TimeChunk("lorem ipsum", start_time=0.0, end_time=7.6),
            TimeChunk("dolor sit amet", start_time=7.7, end_time=10.0),
        ]
        slide_chunks = [
            TimeChunk("lorem", start_time=0.0, end_time=4.9),
            TimeChunk("ipsum", start_time=5.0, end_time=7.5),
            TimeChunk("dolor sit amet", start_time=7.6, end_time=9.9),
        ]

        sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)
        # Looking only at 'lorem ipsum'
        diff = 1.0 - sim_scores[0]

        assert diff < self.error_range

    def test_given_transcript_times_within_slide_time_then_similar(self):
        transcript_chunks = [
            TimeChunk("lorem", start_time=0.0, end_time=4.9),
            TimeChunk("ipsum", start_time=5.0, end_time=10.0),
        ]
        slide_chunks = [
            TimeChunk("lorem", start_time=0.0, end_time=10.0),
        ]

        sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)
        diff = 1.0 - sim_scores[0]

        assert diff < self.error_range

    def test_given_slide_time_overlaps_between_transcript_chunks_then_similar(self):
        transcript_chunks = [TimeChunk("lorem ipsum", start_time=0.0, end_time=7.0)]
        slide_chunks = [
            TimeChunk("lorem", start_time=0.0, end_time=5.0),
            TimeChunk("ipsum", start_time=5.0, end_time=10.0),
        ]

        sim_scores = get_similarity_scores(transcript_chunks, slide_chunks, overlap=1)
        diff = 1.0 - sim_scores[0]

        assert diff < self.error_range

    def test_given_slide_time_no_overlap_between_transcript_chunks_then_not_similar(
        self,
    ):
        transcript_chunks = [TimeChunk("lorem ipsum", start_time=0.0, end_time=7.0)]
        slide_chunks = [
            TimeChunk("lorem", start_time=0.0, end_time=5.0),
            TimeChunk("ipsum", start_time=5.0, end_time=10.0),
        ]

        sim_scores = get_similarity_scores(transcript_chunks, slide_chunks, overlap=3)
        diff = 1.0 - sim_scores[0]

        assert diff > self.error_range
