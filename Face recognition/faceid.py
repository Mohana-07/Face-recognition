# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty, ObjectProperty
import sqlite3  # For database connection

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.textinput import TextInput
from kivy.core.camera import Camera

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
import numpy as np
import playsound
from datetime import datetime
from kivy.uix.widget import Widget
from passlib.hash import bcrypt
from kivy.uix.filechooser import FileChooserListView

class AdminPage(Screen):

    def open_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                # Read the contents of the file
                file_contents = file.read()
                # Print or process the file contents as needed
                print("File contents:")
                print(file_contents)
        except FileNotFoundError:
            print(f"File not found at path: {file_path}")
        except Exception as e:
            print(f"Error opening file: {e}")
    
    def __init__(self, **kwargs):
        super(AdminPage, self).__init__(**kwargs)
        layout = BoxLayout(orientation="vertical")

        # Create UI elements
        
        add_user_button = Button(text="Add New User", on_press=self.add_user)
        self.logout_button = Button(text="Logout", on_press=self.logout)

        verification_log_button = Button(text="View Verification Log", on_press=self.view_verification_log)
        layout.add_widget(verification_log_button)
        Application_Data="D:\\final\\face_rec_sys\\app\\application_data"
        application_data_button = Button(text="Access Application Data ", on_press=lambda x:self.open_file(Application_Data))
        application_data_button.disabled = True  # Initially disable for security
        layout.add_widget(application_data_button)

        # Add UI elements to layout
        
        
        layout.add_widget(add_user_button)
        layout.add_widget(self.logout_button)

        self.add_widget(layout)

    def add_user(self, instance):
         CamApp().capture_new_user()
    
    def view_verification_log(self, instance):
        try:
            with open("verification_log.txt", "r") as log_file:
                log_contents = log_file.read()
            popup = Popup(title="Verification Log", content=Label(text=log_contents), size_hint=(0.8, 0.8))
            popup.open()
        except FileNotFoundError:
            print("Verification log file not found.")  # You might want to display an error message to the user
    
    
    def logout(self, instance):
        self.manager.current = 'login'

    def show_popup(self, title, message):
        popup = Popup(title=title, content=Label(text=message), size_hint=(None, None), size=(400, 200))
        popup.open()

class AdminLoginScreen(Screen):
    # username = StringProperty('')
    # password = StringProperty('')
    # error_message = StringProperty('')

    def __init__(self, **kwargs):
        super(AdminLoginScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation="vertical")

        username_label = Label(text="Username:")
        layout.add_widget(username_label)
        self.username = TextInput(multiline=False)
        layout.add_widget(self.username)

        password_label = Label(text="Password:")
        layout.add_widget(password_label)
        self.password = TextInput(password=True, multiline=False)
        layout.add_widget(self.password)

        login_button = Button(text="Login", on_press=self.login)
        layout.add_widget(login_button)

        # Add the admin login button to the layout
        admin_login_button = Button(text="Admin Login", size_hint=(0.1, 0.1), pos_hint={'top': 1, 'right': 1})
        admin_login_button.bind(on_press=self.show_admin_login_screen)  # Bind the button press event
        layout.add_widget(admin_login_button)

        self.add_widget(layout)

    def login(self, instance):
        username = self.username.text
        password = self.password.text

        hashed_password = cursor.execute("SELECT password FROM admin_users WHERE username=?", (username,)).fetchone()[0]
        if hashed_password and bcrypt.checkpw(password.encode(), hashed_password.encode()):
            self.manager.current = 'admin'
        else:
            # Show error message
            print("Invalid username or password.")  # Replace with proper error handling
    
    def show_admin_login_screen(self, instance):
        self.manager.current = 'login_admin'

    def show_error_popup(self):
        popup = Popup(title='Login Error', content=Label(text=self.error_message),
                      size_hint=(None, None), size=(400, 200))
        popup.open()

    def is_database_authentication(self):  # Flag to choose authentication method
        # Check if a specific environment variable is set to determine authentication method
        authentication_method = os.getenv("AUTHENTICATION_METHOD")

        if authentication_method == "database":
            return True
        elif authentication_method == "file":
            return False
        else:
        # Default to database authentication if no method is specified
            return True
class MainScreen(Screen):
    add_user_button = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.add_user_button = Button(text="Add User")
        self.add_user_button.disabled = True  # Initialize disabled state
        self.add_user_button.bind(on_press=self.enable_add_user_button)  # Bind button press event
        self.add_widget(self.add_user_button)


    def enable_add_user_button(self):
        self.add_user_button.disabled = False



class CameraWidget(Widget):
    def __init__(self,capture, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)
        self.capture = capture  # Webcam capture instance
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Schedule update function at 30 FPS

        self.img = Image()  # Create an Image widget to display the camera feed
        self.add_widget(self.img)
        # Initialize self.popup here (example with a Button)
        #self.popup_button = Button(text="Open Popup")
        # self.popup_button.bind(on_press=self.open_popup)  # Bind a function to open the popup

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Convert the frame to a texture and assign it to the Image widget
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture

# Build app and layout 
class CamApp(App):
    def show_admin_login_screen(self, instance):
        self.manager.current = 'login_admin'


    def build(self):
        layout = BoxLayout(orientation='vertical')
        # Create layouts for each screen
        login_screen_layout = BoxLayout(orientation='vertical')
        admin_page_layout = BoxLayout(orientation='vertical')
        self.manager = ScreenManager()
        # Create an instance of the AdminLoginScreen
        self.login_screen = AdminLoginScreen(name='login')
        self.manager.add_widget(AdminLoginScreen(name='login_admin'))
        self.manager.add_widget(self.login_screen)
        self.manager.add_widget(AdminPage(name='admin'))
        self.manager.add_widget(MainScreen(name='main'))

        self.login_screen.add_widget(layout.clone())  # Clone the layout instance for AdminLoginScreen
        self.admin_page.add_widget(layout.clone())    # Clone the layout instance for AdminPage
        # Setup video capture device (assuming camera index 0)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)  # Schedule update function at 30 FPS

        # Main layout with background image
        
        # background_image = Image(source="C:\\Users\\MohanaKumariParvatha\\OneDrive - Hanyaa\\Desktop\\background image.jpg", pos_hint={'top': 1})
        # layout.add_widget(background_image)

        # Colorful title for "Face Recognition System"
        title_label = Label(text="Face Recognition System", font_size=38, font_name="D:\\final\\Deutschlands.ttf",  # Replace with your font file path
                            halign='center', valign='top', bold=True, color=(1, 1, 1))  # Reddish color
        layout.add_widget(title_label)

        # Main UI elements below the title
        #self.img1 = Image(size_hint=(1, .8))
        # Camera widget for displaying webcam feed
        self.camera_widget = CameraWidget(capture=self.capture, size_hint=(None,None), size=(250, 250), pos_hint={'center_x': 0.5, 'center_y': 0.5} )
        layout.add_widget(self.camera_widget)
        #layout.add_widget(self.img1)
        self.add_user_button = Button(text="Add User", size_hint=(1,.1))
        self.add_user_button.bind(on_press=self.capture_new_user)

        self.verification = Label(text="Verification Uninitiated", size_hint=(1, .1))

        # Add UI elements to the layout
        
        layout.add_widget(self.add_user_button)

        layout.add_widget(self.verification)

        # Add the admin login button to the layout
        admin_login_button = Button(text="Admin Login", size_hint=(0.1, 0.1), pos_hint={'top': 0, 'right': 1})
        admin_login_button.bind(on_press=self.manager.current_screen.show_admin_login_screen)
        layout.add_widget(admin_login_button)

        # Load tensorflow/keras model (assuming 'siamesemodelv2.h5' is your model file)
        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist': L1Dist})

        # Add layouts to the respective screens
        self.login_screen.add_widget(login_screen_layout)
        self.admin_page.add_widget(admin_page_layout)

        # Add a boolean flag to track "Add User" mode
        self.add_user_mode = False
        self.popup = None

        return layout
    
    def on_stop(self):
        # Release video capture device when the application stops
        self.capture.release()
        super().on_stop()

    # Function to send security alert (replace with your preferred method)
    def send_security_alert(self):
        # Example using print statement (replace with actual security system integration)
        print("Security Alert: Unauthorized access detected!")
    
    def capture_new_user(self, instance):
        self.add_user_mode = True  # Set flag to disable verification
        
        # Open popup window for user instructions and image capture
        self.open_add_user_popup()
    
    def stop_adding_users(self, instance):
        self.add_user_popup.dismiss()
        self.add_user_mode = False
        # Trigger a verification cycle to immediately check for faces
        # self.update(None)
    
    # Add method to open admin login screen
    def open_admin_login_screen(self, instance):
        self.login_screen.manager.current = 'login_admin'

    import os  # For file path handling

    def open_add_user_popup(self):
        # Create popup layout with instructions and capture button
        popup_layout = BoxLayout(orientation='vertical')
        popup_label = Label(text="Capture User's Face:")

        # Create a horizontal layout to hold the buttons side-by-side
        button_box = BoxLayout(orientation='horizontal')
        capture_button = Button(text="Capture Image", size_hint=(0.4, 0.1))
        done_button = Button(text="Done", size_hint=(0.4, 0.1), valign='bottom')
        button_box.add_widget(capture_button)
        button_box.add_widget(done_button)

        # Frame for capturing the face (assuming you have webcam access)
        #capture_frame_layout = CameraWidget(size_hint=(1, 0.5))  # Create a layout to hold the Camera widget
        #capture_frame = Camera(size_hint=(1, 1))  # Adjust size_hint as needed
        #capture_frame_layout.add_widget(capture_frame) 

        
        # Bind buttons and add widgets to the layout
        capture_button.bind(on_press=self.capture_and_save_user_image)  # Pass self for access
        done_button.bind(on_press=self.stop_adding_users)  # Assuming stop_adding_users is defined elsewhere
        popup_layout.add_widget(popup_label)
        #popup_layout.add_widget(self.camera_widget)
        popup_layout.add_widget(button_box)

        # Create and show the popup
        self.add_user_popup = Popup(title="Add New User", content=popup_layout, size_hint=(0.8, 0.5))
        self.add_user_popup.open()

    def capture_and_save_user_image(self, instance):

        import datetime
        ret, frame = self.capture.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            return

        if ret:
            # Resize frame to desired dimensions (250x250)
            frame = cv2.resize(frame, (250, 250))

            # Get unique filename based on timestamp for user_name (optional)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            image_name = f"{timestamp}_new_user.jpg"

            # Save the image to verification_images folder
            image_path = os.path.join('application_data', 'verification_images', image_name)
            cv2.imwrite(image_path, frame)

            print(f"New user image saved: {image_path}")
        # Play sound to indicate capture (optional)
        #playsound.playsound("D:\\final\\iphone-camera-capture-6448.wav")  # Replace with your sound file path

        # Close the popup after saving
        #self.popup.dismiss()


    # Run continuously to get webcam feed
    def update(self, *args):
        # if self.popup:
        #     self.popup.dismiss()
        # Read frame from opencv
        ret, frame = self.capture.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            return
        frame = frame[120:120+250, 200:200+250, :]
         # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Check for motion and perform verification only if not in "Add User" mode
        # if is_motion and not self.add_user_mode:
        #     self.prev_frame = gray  # Update previous frame before verification
        #     self.capture_and_verify(frame)  # Call verification function
        if not self.add_user_mode:
            # Handle first frame initialization
            if not hasattr(self, 'prev_frame'):
                self.prev_frame = gray
                return

            # Calculate difference between frames
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]

            # Detect motion using contour area
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            is_motion = any(cv2.contourArea(c) > 1000 for c in contours)

            # Capture image and perform verification on motion detection
            if not self.add_user_mode and is_motion :
                self.prev_frame = gray
                self.capture_and_verify(frame)  # Call a new function for verification logic

        # Update previous frame for next iteration
        self.prev_frame = gray

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # self.web_cam.texture = img_texture
        self.camera_widget.img.texture = img_texture
    


    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img
    def capture_and_verify(self, frame):
        base_path = os.path.join('application_data', 'input_image')
        os.makedirs(base_path, exist_ok=True)  # Ensure input_image folder exists

        image_count = len(os.listdir(base_path)) + 1  # Get current number of images in the folder
        image_name = f"input_image{image_count}.jpg"  # Generate a unique name
        image_save_path = os.path.join(base_path, image_name)

        if frame is not None:
            frame = cv2.resize(frame, (250, 250))
            cv2.imwrite(image_save_path, frame)  # Save the captured image with the unique name

            # Call pre-processing and verification functions
            #self.preprocess(image_save_path)
            self.verify(frame, image_save_path)
        else:
            Logger.warning("Failed to capture frame for verification.")
        ## Create folder to save images after verification (if needed)
        #os.makedirs('verified_images', exist_ok=True)
        
    
    # def dismiss_popup(self, dt):
    #     try:
    #         self.popup.dismiss()
    #     except AttributeError:  # Handle cases where popup might not exist
    #         pass

    # Verification function to verify person
    def verify(self, frame, image_path, *args):  
        import datetime
        # Specify thresholds
        detection_threshold = 0.99
        verification_threshold = 0.8

        # # Capture input image from our webcam
        # SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        # ret, frame = self.capture.read()
        #frame = frame[120:120+250, 200:200+250, :]
        # cv2.imwrite(SAVE_PATH, frame)
        # Dismiss the existing popup if it's open
        if self.popup:
            self.popup.dismiss()

        # Load preprocessed input image
        input_img = self.preprocess(image_path)
        if input_img is None:
            Logger.error("Failed to preprocess input image.")
            return

        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            if validation_img is None:
                Logger.error(f"Failed to preprocess validation image {image}.")
                continue
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        # Set verification text and color based on result 
        self.verification.text = 'Verified' if verified else 'Unverified'
        self.verification.text_color = (0, 1, 0) if verified else (1, 0, 0)  # Green for verified, red for unverified

        # Capture timestamp once (assuming 'frame' is still available)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if verified:
            # Access granted
            welcome_message = f"Welcome! You have been verified at {timestamp}"
            print(welcome_message)  # Replace with your access granting mechanism

            # Save image with timestamp
            image_save_path = os.path.join('verified_images', f"{timestamp}_verified.jpg")
            try:
                cv2.imwrite(image_save_path, frame)
            except Exception as e:
                print(f"Error saving image: {e}")
            

        
        else:
            self.popup = Popup(title="Access Denied", content=Label(text="Verification failed. Access denied."), size_hint=(0.5, 0.1))
            self.popup.open()
            # Ensure popup is defined in a shared scope
            # if not hasattr(self, 'popup'):
            #     self.popup = Popup(title="Access Denied", content=Label(text="Verification failed. Access denied."), size_hint=(0.5, 0.1))
            # Open the popup
            # Immediately close the popup after opening (less efficient approach)
            # self.popup.dismiss()

        # Implement audio alerts using playsound (assuming playsound is installed)
        alert_sound = 'D:\\final\\verified.mp3' if verified else "D:\\final\\unknown_user.wav"
        try:
            playsound.playsound(alert_sound, block=False)  # Play sound asynchronously
        except Exception as e:
            print(f"Error playing sound: {e}")  # Handle potential errors gracefully
        
        base_path = os.path.join('application_data', 'input_image')
        image_count = len(os.listdir(base_path)) + 1  # Get current number of images in the folder
        image_name = f"input_image{image_count}.jpg"
        # Implement logging to file or database
        image_path = os.path.join('application_data', 'input_image', image_name)

        # Example logging to a text file (replace with your preferred logging method)
        with open('verification_log.txt', 'a') as log_file:
            log_file.write(f"{timestamp} - {image_path} - {self.verification.text}\n")
        

        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        
        return results, verified
# Create database connection
conn = sqlite3.connect('admin_db.db')  # Replace with your database file
cursor = conn.cursor()

# Check if admin_users table exists, if not, create it
cursor.execute('''CREATE TABLE IF NOT EXISTS admin_users (
                  username TEXT PRIMARY KEY,
                  password TEXT NOT NULL)''')

# Default admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123@123"  # Replace with a strong password



# if __name__ == '__main__':
#     CamApp.run()
if __name__ == '__main__':
    app = CamApp()
    app.run()
