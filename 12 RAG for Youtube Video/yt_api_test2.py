import youtube_transcript_api as ytt
print(ytt.__file__)

ytt_api = ytt.YouTubeTranscriptApi()

# retrieve the available transcripts
transcript_list = ytt_api.list('Gfr50f6ZBvo')

transcript = transcript_list.find_transcript(['en'])

print(transcript) # en ("English (auto-generated)")[TRANSLATABLE]

text = transcript.fetch()

print(len(text)) # 3790 chunks

print(' '.join(i.text for i in text[0:5])) #the following is a conversation with demus hasabis ceo and co-founder of deepmind a company that has published and builds some of the most incredible artificial