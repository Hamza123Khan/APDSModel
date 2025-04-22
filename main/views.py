import ipaddress
from urllib.parse import urlparse
from rest_framework import viewsets
from rest_framework import status
from rest_framework.response import Response
import re
import tldextract
import pandas as pd
import numpy as np
import joblib
from rest_framework.permissions import IsAuthenticated, AllowAny
from .models import Attachments, Category, Emails, FAQs, Links, ReportAttributes, Reports
from .serializers import (
    AttachmentsSerializer, CategorySerializer, EmailsSerializer, FAQsSerializer,
    LinksSerializer, ReportAttributesSerializer, ReportsSerializer
)
from rest_framework.decorators import action
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated  # Ensure the user is authenticated
from django.conf import settings


# **Emails ViewSet** - Handles CRUD operations for Emails
class EmailsViewSet(viewsets.ModelViewSet):
    serializer_class = EmailsSerializer
    permission_classes = [IsAuthenticated]

    # Get all emails for the logged-in user
    def get_queryset(self):
        return Emails.objects.filter(user_id=self.request.user)

    # Create a new email with attachments and links
    def create(self, request, *args, **kwargs):
        
        email_data = request.data
        attachments_data = email_data.pop('attachments', [])
        links_data = email_data.pop('links', [])
        
        # Set the user_id field to the authenticated user
        email_data['user_id'] = request.user.id

        email_serializer = self.get_serializer(data=email_data)
        email_serializer.is_valid(raise_exception=True)
        email = email_serializer.save(user_id=request.user)

        # Handle attachments
        for attachment in attachments_data:
            attachment['email_id'] = email.id
            attachment_serializer = AttachmentsSerializer(data=attachment)
            attachment_serializer.is_valid(raise_exception=True)
            attachment_serializer.save()

        # Handle links
        for link in links_data:
            link['email_id'] = email.id
            link_serializer = LinksSerializer(data=link)
            link_serializer.is_valid(raise_exception=True)
            link_serializer.save()

        return Response(email_serializer.data, status=status.HTTP_201_CREATED)

    # Update email with attachments and links
    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        email_data = request.data
        attachments_data = email_data.pop('attachments', [])
        links_data = email_data.pop('links', [])

        email_serializer = self.get_serializer(instance, data=email_data, partial=True)
        email_serializer.is_valid(raise_exception=True)
        email_serializer.save()

        # Handle attachments
        for attachment in attachments_data:
            attachment['email_id'] = instance.id
            attachment_serializer = AttachmentsSerializer(data=attachment)
            attachment_serializer.is_valid(raise_exception=True)
            attachment_serializer.save()

        # Handle links
        for link in links_data:
            link['email_id'] = instance.id
            link_serializer = LinksSerializer(data=link)
            link_serializer.is_valid(raise_exception=True)
            link_serializer.save()

        return Response(email_serializer.data)

    # Delete email and related attachments and links
    def destroy(self, request, *args, **kwargs):
        email = self.get_object()
        email_id = email.id

        # Delete related attachments and links
        Attachments.objects.filter(email_id=email_id).delete()
        Links.objects.filter(email_id=email_id).delete()

        return super().destroy(request, *args, **kwargs)


# **Reports ViewSet** - Handles CRUD operations for Reports
class ReportsViewSet(viewsets.ModelViewSet):
    serializer_class = ReportsSerializer
    permission_classes = [IsAuthenticated]

    # Get reports filtered by email_id (requesting user's reports)
    def get_queryset(self):
        email_id = self.request.query_params.get('email_id')
        if email_id:
            return Reports.objects.filter(email_id=email_id)
        return Reports.objects.none()

    # Create a new report with associated reportAttributes
    def create(self, request, *args, **kwargs):
        report_data = request.data
        report_attributes_data = report_data.pop('attributes', [])

        report_serializer = self.get_serializer(data=report_data)
        report_serializer.is_valid(raise_exception=True)
        report = report_serializer.save()

        # Handle reportAttributes
        for attribute in report_attributes_data:
            attribute['report_id'] = report.id
            attribute_serializer = ReportAttributesSerializer(data=attribute)
            attribute_serializer.is_valid(raise_exception=True)
            attribute_serializer.save()

        return Response(report_serializer.data, status=status.HTTP_201_CREATED)

    # Update report with reportAttributes
    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        report_data = request.data
        report_attributes_data = report_data.pop('attributes', [])

        report_serializer = self.get_serializer(instance, data=report_data, partial=True)
        report_serializer.is_valid(raise_exception=True)
        report_serializer.save()

        # Handle reportAttributes
        for attribute in report_attributes_data:
            attribute['report_id'] = instance.id
            attribute_serializer = ReportAttributesSerializer(data=attribute)
            attribute_serializer.is_valid(raise_exception=True)
            attribute_serializer.save()

        return Response(report_serializer.data)

    # Delete report and related reportAttributes
    def destroy(self, request, *args, **kwargs):
        report = self.get_object()
        report_id = report.id

        # Delete related reportAttributes
        ReportAttributes.objects.filter(report_id=report_id).delete()

        return super().destroy(request, *args, **kwargs)


# **FAQs ViewSet** - Handles CRUD operations for FAQs (accessible by anyone)
class FAQsViewSet(viewsets.ModelViewSet):
    queryset = FAQs.objects.all()
    serializer_class = FAQsSerializer
    permission_classes = [AllowAny]  # Open access to FAQs
    
    # Permission control based on HTTP method
    def get_permissions(self):
        if self.action == 'list' or self.action == 'retrieve':
            # Public access for GET (list and retrieve) actions
            return [AllowAny()]
        return [IsAuthenticated()]  # Authenticated access for POST, PUT, DELETE


# **Attachments ViewSet** - Handles CRUD operations for Attachments
class AttachmentsViewSet(viewsets.ModelViewSet):
    queryset = Attachments.objects.all()
    serializer_class = AttachmentsSerializer
    permission_classes = [IsAuthenticated]


# **Category ViewSet** - Handles CRUD operations for Categories
class CategoryViewSet(viewsets.ModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    permission_classes = [IsAuthenticated]


# **Links ViewSet** - Handles CRUD operations for Links
class LinksViewSet(viewsets.ModelViewSet):
    queryset = Links.objects.all()
    serializer_class = LinksSerializer
    permission_classes = [IsAuthenticated]


# **ReportAttributes ViewSet** - Handles CRUD operations for ReportAttributes
class ReportAttributesViewSet(viewsets.ModelViewSet):
    queryset = ReportAttributes.objects.all()
    serializer_class = ReportAttributesSerializer
    permission_classes = [IsAuthenticated]



class SpamClassifierView(APIView):
    model = None
    tokenizer = None
    model_path = None
    tokenizer_path = None
    url_model = None
    url_model_path = None
    selector = None
    clf = None
    feature_columns = None

    permission_classes = [IsAuthenticated]  # Add this to require authentication
    # permission_classes = [AllowAny]  # Add this to require authentication

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define paths for model and tokenizer
        self.model_path = os.path.join(settings.BASE_DIR, 'main\\trainedModelFiles\\lstm_model.h5')
        self.tokenizer_path = os.path.join(settings.BASE_DIR, 'main\\trainedModelFiles\\tokenizer.json')
        self.url_model_path = os.path.join(settings.BASE_DIR, 'main\\trainedModelFiles\\phishing_model.pkl')
        self.selector_path = os.path.join(settings.BASE_DIR, 'main\\trainedModelFiles\\feature_selector.pkl')
        self.feature_coloumns_path = os.path.join(settings.BASE_DIR, 'main\\trainedModelFiles\\feature_columns.pkl')
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        try:
            # Load the trained LSTM model
            self.model = load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
            print(self.model.summary)
        except Exception as e:
            print(f"Error loading model: {e}")

        try:
            # Load the tokenizer from the JSON file
            with open(self.tokenizer_path, 'r') as json_file:
                tokenizer_json = json.load(json_file)
            self.tokenizer = tokenizer_from_json(tokenizer_json)
            for key, values in tokenizer_json:
                print(f"{key}:{values[-1]}")
            print(f"Tokenizer loaded from {self.tokenizer_path}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
        try:
            self.url_model = joblib.load(self.url_model_path)
            print(f"URL Model loaded from {self.url_model_path}")
        except Exception as e:
            print(f"Error loading URL model: {e}")
        try:
            self.url_model = joblib.load(self.url_model_path)
            self.selector = joblib.load(self.selector_path)
            self.feature_columns = joblib.load(self.feature_coloumns_path)
            print("additional components(classifies, feature coloumns, selector)")
        except Exception as e:
            print("error loading components",e)

    def preprocessing(self, email_content):
        """Preprocess email content for prediction."""
        test_sequences = self.tokenizer.texts_to_sequences([email_content])
        return pad_sequences(test_sequences, padding='post', maxlen=100)
    def count_special_chars(self,url):
        """Count special characters in URL."""
        return len(re.findall(r'[./?\-_@=]', url))
    def has_ip(self,url):
        """Check if URL contains an IP address."""
        try:
            ipaddress.ip_address(urlparse(url).netloc)
            return 1
        except ValueError:
            return 0
    def ExtractFeatures(self, url):
        features = {}

        # Extracting features correctly
        features['url_length'] = len(url)  #  This should be a number, not zero!
        features['num_special_chars'] = self.count_special_chars(url)  #  Count special characters
        features['num_digits'] = sum(c.isdigit() for c in url)  #  Count digits
        features['num_letters'] = sum(c.isalpha() for c in url)  #  Count letters
        features['num_subdomains'] = len(tldextract.extract(url).subdomain.split('.'))  #  Count subdomains
        features['has_ip'] = self.has_ip(url)  #  Check if URL contains an IP
        features['uses_https'] = 1 if urlparse(url).scheme == 'https' else 0  #  HTTPS check

        # Extract Top-Level Domain (TLD)
        tld = tldextract.extract(url).suffix
        top_tlds = ['com', 'org', 'net', 'edu', 'gov', 'co', 'info', 'biz', 'xyz', 'cn']
        features['tld'] = tld if tld in top_tlds else 'other'

        print("Extracted Raw Features:", features)  #  Debugging Output

        # Convert to DataFrame
        features_df = pd.DataFrame([features])

        # One-hot encoding for TLD
        features_df = pd.get_dummies(features_df, columns=['tld'], drop_first=True)

        # Align with training features
        features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)

        print("Final Processed Features:", features_df)  # Debugging Output

        return features_df
    # def get(self, request, *args, **kwargs):
    #     """Handle the GET request and return a sample string."""
    #     # sample_string = "Congratulations! You WON the Lottery!!!."
    #     sample_string = "How are you doing today? are you available for a call today?."
    #     processed_content = self.preprocessing(sample_string)
    #     # Predict with the model
    #     prediction_result = self.model.predict(processed_content)
    #     # Determine if the email is spam or legitimate
    #     result = "Spam Email" if prediction_result[0] > 0.5 else "Legitimate Email"
    #     return Response({'Result':result}, status=status.HTTP_200_OK)
    
    def post(self, request, *args, **kwargs):
        """Handle the POST request for spam classification."""
        result = {"Email classification": "unknown", "Url_Classification": "unknown"}
        
        if request.method == "POST":
            # Get email content and URL content
            mail_content = request.data.get('email_content')
            Url_content = request.data.get('Url_content') or request.data.get('url_content')  # Case insensitive

            print("Debug: Received email content:", mail_content)
            print("Debug: Received URL content:", Url_content)

            if mail_content:
                processed_content = self.preprocessing(mail_content)
                prediction_result = self.model.predict(processed_content)
                result["Email classification"] = "Spam Email" if prediction_result[0] > 0.5 else "Legitimate Email"
            
            if Url_content:  # Changed from `elif` to `if` to allow processing both
                print("Processing URL content:", Url_content)
                test_features = self.ExtractFeatures(Url_content)
                print("Extracted features:", test_features)

                if self.selector is not None:
                    test_features = self.selector.transform(test_features)
                    print("Transformed features:", test_features)

                if self.url_model is not None:
                    predictions = self.url_model.predict(test_features)
                    result["Url_Classification"] = "Phishing URL" if predictions[0] == 1 else "Legitimate URL"
                else:
                    print("Warning: URL Model is not loaded!")

        return Response({"result": result})
