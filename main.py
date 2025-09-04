from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from mediapipe.python.solutions import pose as mp_pose
from PIL import Image, ImageDraw
import io
import json
import requests
import base64 # For encoding image to send to Stable Diffusion API

# Supabase Client
from supabase import create_client, Client

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Supabase Initialization (from environment variables) ---
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # Use Service Role Key for backend operations
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Stable Diffusion API Endpoint (from environment variables) ---
SD_API_ENDPOINT: str = os.environ.get('SD_API_ENDPOINT', 'YOUR_STABLE_DIFFUSION_API_ENDPOINT')
SD_API_KEY: str = os.environ.get('SD_API_KEY', 'YOUR_STABLE_DIFFUSION_API_KEY') # If your SD API needs a key

# --- BodyAnalyzer and SketchGenerator classes remain mostly the same ---
# (Copy these directly from the previous response)

class BodyAnalyzer:
    def __init__(self):
        self.mp_pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.pose_landmarks = mp_pose.PoseLandmark

    def _calculate_distance_pixels(self, p1, p2, image_width, image_height):
        x1, y1 = p1.x * image_width, p1.y * image_height
        x2, y2 = p2.x * image_width, p2.y * image_height
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def analyze_body(self, image_bytes, age_group='adult'): # Accepts image bytes directly
        # Decode image from bytes
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image bytes.")
        
        max_dim = 800
        h, w, _ = image.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(image_rgb)

        measurements = {}
        body_type = "unknown"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            img_h, img_w, _ = image.shape

            left_shoulder = landmarks[self.pose_landmarks.LEFT_SHOULDER]
            right_shoulder = landmarks[self.pose_landmarks.RIGHT_SHOULDER]
            measurements['shoulders_width_ratio'] = self._calculate_distance_pixels(left_shoulder, right_shoulder, img_w, img_h) / img_h

            left_hip = landmarks[self.pose_landmarks.LEFT_HIP]
            right_hip = landmarks[self.pose_landmarks.RIGHT_HIP]
            measurements['hips_width_ratio'] = self._calculate_distance_pixels(left_hip, right_hip, img_w, img_h) / img_h
            
            measurements['waist_width_ratio'] = measurements['hips_width_ratio'] * 0.85 # Heuristic

            left_arm_length = self._calculate_distance_pixels(left_shoulder, landmarks[self.pose_landmarks.LEFT_WRIST], img_w, img_h)
            right_arm_length = self._calculate_distance_pixels(right_shoulder, landmarks[self.pose_landmarks.RIGHT_WRIST], img_w, img_h)
            measurements['arms_length_ratio'] = (left_arm_length + right_arm_length) / (2 * img_h)

            left_hip_lm = landmarks[self.pose_landmarks.LEFT_HIP] # Re-get as landmark objects
            right_hip_lm = landmarks[self.pose_landmarks.RIGHT_HIP]
            left_leg_length = self._calculate_distance_pixels(left_hip_lm, landmarks[self.pose_landmarks.LEFT_ANKLE], img_w, img_h)
            right_leg_length = self._calculate_distance_pixels(right_hip_lm, landmarks[self.pose_landmarks.RIGHT_ANKLE], img_w, img_h)
            measurements['legs_length_ratio'] = (left_leg_length + right_leg_length) / (2 * img_h)
            
            nose = landmarks[self.pose_landmarks.NOSE]
            left_foot = landmarks[self.pose_landmarks.LEFT_FOOT_INDEX]
            right_foot = landmarks[self.pose_landmarks.RIGHT_FOOT_INDEX]
            lowest_foot_y = max(left_foot.y, right_foot.y)
            measurements['relative_height'] = (lowest_foot_y - nose.y) if lowest_foot_y > nose.y else 0.0

            # --- Age-Group Specific Adjustments and Body Type ---
            if age_group == 'child':
                body_type = "child_average_build"
                if measurements['shoulders_width_ratio'] < 0.2: body_type = "child_slim_build"
                elif measurements['shoulders_width_ratio'] > 0.35: body_type = "child_sturdy_build"
                measurements['waist_width_ratio'] *= 1.1 
            
            elif age_group == 'elderly':
                shoulder_to_hip_ratio = measurements['shoulders_width_ratio'] / measurements['hips_width_ratio'] if measurements['hips_width_ratio'] else 1
                body_type = "elderly_rectangle"

                if shoulder_to_hip_ratio > 1.15: body_type = "elderly_inverted_triangle"
                elif shoulder_to_hip_ratio < 0.85: body_type = "elderly_pear"
                elif measurements['waist_width_ratio'] > measurements['hips_width_ratio'] * 0.9: body_type = "elderly_apple"
                elif measurements['waist_width_ratio'] < measurements['hips_width_ratio'] * 0.8: body_type = "elderly_hourglass"
                
                measurements['waist_width_ratio'] *= 1.05
                measurements['hips_width_ratio'] *= 1.02
                
            else: # 'adult' or default
                shoulder_to_hip_ratio = measurements['shoulders_width_ratio'] / measurements['hips_width_ratio'] if measurements['hips_width_ratio'] else 1

                if shoulder_to_hip_ratio > 1.2: body_type = "inverted triangle"
                elif shoulder_to_hip_ratio < 0.8: body_type = "pear"
                elif 0.9 < shoulder_to_hip_ratio < 1.1 and measurements['waist_width_ratio'] > measurements['hips_width_ratio'] * 0.95:
                    body_type = "apple"
                elif 0.9 < shoulder_to_hip_ratio < 1.1 and measurements['waist_width_ratio'] < measurements['hips_width_ratio'] * 0.8:
                    body_type = "hourglass" if abs(measurements['shoulders_width_ratio'] - measurements['hips_width_ratio']) < 0.1 * measurements['shoulders_width_ratio'] else "rectangle"
                else: body_type = "rectangle"

        return measurements, body_type

class SketchGenerator:
    def __init__(self, sd_api_endpoint, sd_api_key):
        self.sd_api_endpoint = sd_api_endpoint
        self.sd_api_key = sd_api_key # For potential API authentication

    def generate_prompt(self, body_type, measurements, style, occasion, color, fabric=None, age_group='adult'):
        age_descriptor = ""
        if age_group == 'child':
            age_descriptor = "child-like figure, young, cute, "
            body_type_desc = body_type.replace("child_", "").replace("_build", "") + " build"
            prompt = f"line art outline sketch of a {age_descriptor}{body_type_desc}, "
            prompt += f"wearing a {color} colored, comfortable {style} outfit, suitable for {occasion}. "
            if fabric:
                prompt += f"Made of {fabric}. "
            prompt += "minimal shading, simple, flat, clean fashion design sketch, full body, white background."
        elif age_group == 'elderly':
            age_descriptor = "elderly person, graceful, "
            body_type_desc = body_type.replace("elderly_", "")
            prompt = f"line art outline sketch of an {age_descriptor}{body_type_desc} body type, "
            prompt += f"wearing a {color} colored, elegant {style} outfit, suitable for {occasion}. "
            if fabric:
                prompt += f"Made of {fabric}. "
            prompt += "comfortable, loose fit, minimal shading, professional fashion design sketch, full body, white background."
        else: # Adult
            body_type_desc = body_type
            prompt = f"line art outline sketch of a person with {body_type_desc}, "
            prompt += f"wearing a {color} colored {style} outfit, suitable for a {occasion}. "
            if fabric:
                prompt += f"Made of {fabric}. "
            prompt += "minimal shading, professional fashion design sketch, full body, white background."

        negative_prompt = "blurry, low resolution, extra limbs, bad anatomy, text, watermark, cartoon, painting, unrealistic, photorealistic, deformed, distorted, ugly, pixelated, bad quality"
        return prompt, negative_prompt

    def generate_sketch(self, prompt, negative_prompt, control_image_bytes=None):
        # --- REPLACE WITH ACTUAL STABLE DIFFUSION / CONTROLNET API CALL ---
        # This is a MOCK function.
        # Example with `requests`:
        # try:
        #     payload = {
        #         "prompt": prompt,
        #         "negative_prompt": negative_prompt,
        #         "steps": 20,
        #         "width": 512,
        #         "height": 768,
        #         "control_image": base64.b64encode(control_image_bytes).decode('utf-8') if control_image_bytes else None,
        #         "control_type": "openpose"
        #     }
        #     headers = {"Authorization": f"Bearer {self.sd_api_key}", "Content-Type": "application/json"}
        #     response = requests.post(self.sd_api_endpoint + "/generate", json=payload, headers=headers, timeout=120)
        #     response.raise_for_status()
        #     return base64.b64decode(response.json()['image_base64'])
        # except Exception as e:
        #     print(f"Error calling Stable Diffusion API: {e}")
        #     # Fallback to dummy image on API failure
        #     pass

        print(f"MOCK Sketch Generation for prompt: {prompt}")
        
        # Placeholder image generation for demonstration
        dummy_image = Image.new('RGB', (512, 768), color = 'white')
        d = ImageDraw.Draw(dummy_image)
        
        d.line([(256, 100), (256, 200)], fill="black", width=5)
        d.line([(200, 200), (312, 200)], fill="black", width=5)
        d.line([(256, 200), (256, 400)], fill="black", width=5)
        d.line([(256, 400), (200, 600)], fill="black", width=5)
        d.line([(256, 400), (312, 600)], fill="black", width=5)
        d.line([(200, 200), (150, 300)], fill="black", width=5)
        d.line([(312, 200), (362, 300)], fill="black", width=5)

        if "child" in prompt:
            d.ellipse((230, 80, 280, 130), outline="red", width=3)
            d.text((200, 650), "CHILD FIT", fill="blue")
        elif "elderly" in prompt:
            d.line([(240, 180), (270, 180)], fill="gray", width=2)
            d.text((200, 650), "ELDERLY FIT", fill="green")
        else:
            d.text((200, 650), "ADULT FIT", fill="black")
            
        img_byte_arr = io.BytesIO()
        dummy_image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()


# --- Main AI Processing Endpoint ---
@app.route('/process_order_ai', methods=['POST'])
def process_order_ai_endpoint():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        user_id = data.get('userId')
        order_id = data.get('orderId')
        photo_url_path = data.get('photoUrlPath') # Supabase Storage path
        style_preferences = data.get('stylePreferences', {})
        age_group_pref = data.get('ageGroup', 'adult')

        if not all([user_id, order_id, photo_url_path]):
            return jsonify({"error": "Missing required data (userId, orderId, photoUrlPath)"}), 400

        print(f"Processing AI for Order: {order_id}, User: {user_id}, Photo Path: {photo_url_path}")

        # 1. Download image from Supabase Storage
        try:
            # photo_url_path example: 'customer_uploads/user_uuid/order_uuid/photo.jpg'
            bucket_name = photo_url_path.split('/')[0] # Assuming first part is bucket name (e.g., 'customer_uploads')
            object_path = '/'.join(photo_url_path.split('/')[1:]) # The rest is the object path
            
            res_download = supabase.storage.from_(bucket_name).download(object_path)
            if res_download.data:
                image_bytes = res_download.data
            else:
                raise Exception(f"Failed to download image from Supabase Storage: {res_download.error}")
        except Exception as e:
            print(f"Error downloading image from Supabase: {e}")
            # Update order status in Supabase DB (using the Supabase client here directly)
            supabase.from_('orders').update({
                'processing_status': 'error_downloading_image',
                'last_error': str(e),
                'status': 'failed_ai_processing',
                'updated_at': 'now()' # Use Supabase function for timestamp
            }).eq('id', order_id).execute()
            return jsonify({"error": f"Failed to download image: {e}"}), 500

        # 2. Body Analysis
        body_analyzer = BodyAnalyzer()
        measurements, body_type = body_analyzer.analyze_body(image_bytes, age_group=age_group_pref)
        print(f"Detected Age Group: {age_group_pref}, Body Type: {body_type}, Measurements: {measurements}")

        # 3. Generate AI Sketch
        sketch_generator = SketchGenerator(SD_API_ENDPOINT, SD_API_KEY)
        prompt, negative_prompt = sketch_generator.generate_prompt(
            body_type=body_type,
            measurements=measurements,
            style=style_preferences.get('style', 'casual'),
            occasion=style_preferences.get('occasion', 'everyday'),
            color=style_preferences.get('color', 'blue'),
            fabric=style_preferences.get('fabric'),
            age_group=age_group_pref
        )
        sketch_image_bytes = sketch_generator.generate_sketch(prompt, negative_prompt, control_image_bytes=image_bytes)

        # 4. Upload Sketch to Supabase Storage
        sketch_bucket = 'sketches' # Assuming a 'sketches' bucket in Supabase Storage
        sketch_object_path = f"{user_id}/{order_id}_sketch.png"
        
        res_upload = supabase.storage.from_(sketch_bucket).upload(
            file=sketch_image_bytes,
            path=sketch_object_path,
            file_options={"content-type": "image/png"}
        )

        if res_upload.error:
            raise Exception(f"Failed to upload sketch to Supabase Storage: {res_upload.error.message}")
        
        # Construct public URL (adjust based on your Supabase Storage settings)
        sketch_url = f"{SUPABASE_URL}/storage/v1/object/public/{sketch_bucket}/{sketch_object_path}"
        print(f"Sketch uploaded to: {sketch_url}")

        # 5. Update Order in Supabase DB
        update_data = {
            'body_measurements': measurements,
            'body_type': body_type,
            'ai_prompt': prompt,
            'sketch_url': sketch_url,
            'processing_status': 'sketch_generated',
            'status': 'sketch_generated',
            'updated_at': 'now()'
        }
        res_update = supabase.from_('orders').update(update_data).eq('id', order_id).execute()

        if res_update.error:
            raise Exception(f"Failed to update order in Supabase DB: {res_update.error.message}")

        return jsonify({
            "status": "success",
            "orderId": order_id,
            "sketchUrl": sketch_url,
            "bodyMeasurements": measurements,
            "bodyType": body_type
        }), 200

    except Exception as e:
        print(f"Critical error in AI processing endpoint: {e}")
        # Attempt to log to DB if order_id is available
        order_id_for_error = request.json.get('orderId') if request.json else 'unknown'
        supabase.from_('orders').update({
            'processing_status': 'critical_ai_error',
            'last_error': str(e),
            'status': 'failed_ai_processing',
            'updated_at': 'now()'
        }).eq('id', order_id_for_error).execute()
        return jsonify({"error": str(e)}), 500

# To run this Flask app locally
if __name__ == '__main__':
    # Set dummy environment variables for local testing
    os.environ['SUPABASE_URL'] = 'YOUR_SUPABASE_URL' # e.g., 'https://abcde12345.supabase.co'
    os.environ['SUPABASE_SERVICE_ROLE_KEY'] = 'YOUR_SUPABASE_SERVICE_ROLE_KEY'
    os.environ['SD_API_ENDPOINT'] = 'http://localhost:7860/sdapi/v1' # Or your mock/real endpoint
    os.environ['SD_API_KEY'] = 'YOUR_SD_API_KEY' # If applicable

    app.run(debug=True, host='0.0.0.0', port=5000)
