#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

class RecipeRecommender:
    def __init__(self, k=30, epochs=100, alpha=0.001, beta=0.01, verbose=True):
        self.k = k
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.P = None
        self.Q = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.best_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0
        self.user_map = None
        self.item_map = None
        
    def preprocess_data(self, df):
        """Preprocess the ratings dataframe to create user and item mappings and rating matrix"""
        # Create user and item mappings
        user_ids = df['user_id'].unique()
        item_ids = df['recipe_id'].unique()
        self.user_map = {uid: i for i, uid in enumerate(user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(item_ids)}
        
        # Create rating matrix
        R = np.zeros((len(user_ids), len(item_ids)))
        for _, row in df.iterrows():
            R[self.user_map[row['user_id']], self.item_map[row['recipe_id']]] = row['rating']
        
        return R
    
    def fit(self, ratings_df):
        """Train the recommendation model"""
        # Preprocess data
        R = self.preprocess_data(ratings_df)
        self.users, self.items = R.shape
        
        # Initialize global mean
        known_ratings = R[R > 0]
        self.global_mean = np.mean(known_ratings) if len(known_ratings) > 0 else 0
        
        # Initialize biases
        self.user_bias = np.zeros(self.users)
        self.item_bias = np.zeros(self.items)
        
        # Calculate initial biases
        for i in range(self.users):
            user_ratings = R[i, R[i] > 0]
            if len(user_ratings) > 0:
                self.user_bias[i] = np.mean(user_ratings) - self.global_mean
        
        for j in range(self.items):
            item_ratings = R[R[:, j] > 0, j]
            if len(item_ratings) > 0:
                self.item_bias[j] = np.mean(item_ratings) - self.global_mean
        
        # Initialize latent factors with better scaling
        self.P = np.random.normal(scale=0.1, size=(self.users, self.k))
        self.Q = np.random.normal(scale=0.1, size=(self.items, self.k))
        
        known_indices = np.argwhere(R > 0)
        n_samples = len(known_indices)
        
        # Learning rate schedule
        initial_lr = self.alpha
        min_lr = initial_lr * 0.01
        
        for epoch in range(self.epochs):
            np.random.shuffle(known_indices)
            epoch_loss = 0
            
            # Decay learning rate
            current_lr = max(min_lr, initial_lr * (1.0 / (1.0 + epoch * 0.1)))
            
            for i, j in known_indices:
                # Calculate prediction with bias terms
                pred = (self.global_mean + 
                       self.user_bias[i] + 
                       self.item_bias[j] + 
                       np.dot(self.P[i, :], self.Q[j, :]))
                
                error = R[i, j] - pred
                epoch_loss += error ** 2
                
                # Update latent factors
                self.P[i, :] += current_lr * (error * self.Q[j, :] - self.beta * self.P[i, :])
                self.Q[j, :] += current_lr * (error * self.P[i, :] - self.beta * self.Q[j, :])
                
                # Update bias terms
                self.user_bias[i] += current_lr * (error - self.beta * self.user_bias[i])
                self.item_bias[j] += current_lr * (error - self.beta * self.item_bias[j])
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / n_samples
            
            # Early stopping
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            if self.verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}: MSE = {avg_loss:.4f}")
    
    def predict(self, user_id=None, recipe_id=None):
        """Predict ratings for all user-item pairs or a specific user-item pair"""
        if user_id is not None and recipe_id is not None:
            # Predict for a specific user-item pair
            if user_id in self.user_map and recipe_id in self.item_map:
                u = self.user_map[user_id]
                i = self.item_map[recipe_id]
                return (self.global_mean + 
                        self.user_bias[u] + 
                        self.item_bias[i] + 
                        np.dot(self.P[u, :], self.Q[i, :]))
            else:
                return None
        else:
            # Predict for all user-item pairs
            return (self.global_mean + 
                    self.user_bias[:, np.newaxis] + 
                    self.item_bias[np.newaxis, :] + 
                    np.dot(self.P, self.Q.T))
    
    def evaluate(self, test_df):
        """Evaluate the model on a test set"""
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            if row['user_id'] in self.user_map and row['recipe_id'] in self.item_map:
                pred = self.predict(row['user_id'], row['recipe_id'])
                predictions.append(pred)
                actuals.append(row['rating'])
        
        if len(predictions) == 0:
            return None, None, None
        
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        
        return mse, rmse, mae
    
    def cross_validate(self, df, k=5):
        """Perform k-fold cross-validation"""
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        mse_scores = []
        rmse_scores = []
        mae_scores = []
        
        for train_idx, val_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
 