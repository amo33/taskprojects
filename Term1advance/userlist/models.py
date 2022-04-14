from django.db import models

# Create your models here.

class User(models.Model):
    username = models.TextField()
    age = models.IntegerField()
    Image = models.TextField()

    def __str__(self):
        return self.username