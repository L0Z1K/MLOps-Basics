FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN export LC_ALL=C.UTF-8 && export LANG=C.UTF-8
EXPOSE 8000 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]