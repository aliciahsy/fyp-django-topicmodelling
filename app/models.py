from django.db import models

class Video(models.Model):
    VideoID = models.AutoField(primary_key=True)  # Assuming thr field to be an auto-incrementing numeric ID.
    VideoTitle = models.TextField()
    VideoType = models.TextField()
    VideoURL = models.URLField()
    VideoTranscript = models.FileField(upload_to='transcripts/csv/')  # The csv file will be uploaded to the 'transcripts/csv/' directory within the project.

    def __str__(self):
        return self.VideoTitle
    
class VideoResult(models.Model):
    video = models.OneToOneField(Video, on_delete=models.CASCADE)
    main_topic = models.TextField()
    texts = models.TextField()
    chart_data = models.TextField()
    word_cloud_data = models.TextField()
    pie_chart_data = models.TextField()
    segmented_transcript = models.TextField()
    # Add fields for bag of words (bow_data) and tf-idf (tfidf_data) if needed in next phase