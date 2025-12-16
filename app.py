# --- IMPORTER ---
import cv2
import os
import numpy as np
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

# --- INITIALISER FLASK-APPEN ---
app = Flask(__name__)

# --- GLOBALE VARIABLER OG KONSTANTER ---
DATASET_DIR = "database"
DETECTIONS_DIR = "detections"  # Ny mappe for deteksjonshistorikk
TRAINER_FILE = "trainer.yml"
LOG_FILE = "system.log"

LAST_UNKNOWN_SAVE_TIME = 0
UNKNOWN_SAVE_INTERVAL = 10  # Sekunder mellom lagring av ukjente fjes

# Dict for å spore siste gang en kjent person ble detektert
last_detection_time = {}
DETECTION_COOLDOWN = 5  # Sekunder mellom lagring av samme person

# OpenCV objekter
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = None

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("ADVARSEL: cv2.face modul ikke tilgjengelig. Installér opencv-contrib-python")

id_to_name = {}
model_trained = False

# Opprett mapper hvis de ikke finnes
for directory in [DATASET_DIR, DETECTIONS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- HJELPEFUNKSJONER ---
def log_event(message):
    """Logger hendelser til fil og konsoll"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def load_names_map():
    """Laster inn mapping fra ID til navn basert på filnavn"""
    global id_to_name
    id_to_name = {}
    
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.jpg')]
    name_to_id = {}
    next_id = 1
    
    for file in files:
        if file.startswith("Unknown_"):
            continue
            
        name = file.rsplit('_', 1)[0]
        if name not in name_to_id:
            name_to_id[name] = next_id
            id_to_name[next_id] = name
            next_id += 1
    
    log_event(f"Lastet inn {len(id_to_name)} unike personer")
    return id_to_name

def add_timestamp_to_image(frame, text="", position="top"):
    """Legger til dato/tid og tekst på bildet - kun i hjørnet"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Mindre, diskret timestamp i hjørnet
    if position == "top":
        y_pos = 20
    else:
        y_pos = frame.shape[0] - 10
    
    # Legg til timestamp i øvre høyre hjørne (mindre og mer diskret)
    cv2.putText(frame, timestamp, (frame.shape[1] - 180, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Legg til ekstra tekst hvis oppgitt (under timestamp)
    if text:
        cv2.putText(frame, text, (frame.shape[1] - 180, y_pos + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return frame

def save_detection(face_img, name, confidence=None):
    """Lagrer et detektert ansikt med timestamp - ALLTID i detections/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ALLE deteksjoner lagres i detections/ med tidsstempel
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(DETECTIONS_DIR, filename)
    
    # For Unknown: lagre også i database/ for trening
    if name == "Unknown":
        database_path = os.path.join(DATASET_DIR, filename)
        cv2.imwrite(database_path, face_img)
    
    # Konverter til fargebilde og lagre i detections/
    display_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    
    # Kun timestamp i hjørnet
    time_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(display_img, time_str, (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(filepath, display_img)
    
    return filename

# --- VIDEOSTRØM GENERATOR ---
def gen_frames():
    global LAST_UNKNOWN_SAVE_TIME, model_trained, last_detection_time
    
    camera = cv2.VideoCapture(0)
    
    # Last inn modell hvis den finnes
    if os.path.exists(TRAINER_FILE) and recognizer:
        try:
            recognizer.read(TRAINER_FILE)
            model_trained = True
            log_event("✅ Treningsmodell lastet - klar til gjenkjenning")
        except Exception as e:
            log_event(f"❌ Feil ved innlesing av modell: {e}")
            model_trained = False
    else:
        log_event("⚠️  Ingen treningsmodell - alle ansikter vil være 'Unknown'")
        model_trained = False
    
    load_names_map()

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Legg til timestamp på live-strømmen (kun i hjørnet)
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp_str, (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Forbedret ansiktsdeteksjon - mer stabil
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,      # Mer nøyaktig (tidligere 1.2)
            minNeighbors=6,       # Mer streng (tidligere 5) - reduserer false positives
            minSize=(50, 50),     # Større minimum (tidligere 30x30) - bedre kvalitet
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        current_time = time.time()

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_img = gray[y:y+h, x:x+w]
            
            name = "Unknown"
            confidence = 100.0
            
            # Prøv å gjenkjenne hvis modellen er trent
            if recognizer and id_to_name and model_trained:
                try:
                    id_, conf = recognizer.predict(face_img)
                    confidence = conf
                    
                    # VIKTIG: Lavere confidence = bedre match
                    # Justert grense for mer stabil gjenkjenning
                    if conf < 50:  # Veldig sikker match
                        name = id_to_name.get(id_, "Unknown ID")
                    elif conf < 80:  # Ganske sikker match
                        name = id_to_name.get(id_, "Unknown ID")
                    else:  # For usikker
                        name = "Unknown"
                except:
                    name = "Unknown"

            # Vis navn og confidence på video
            label = f"{name} ({confidence:.1f})"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # LAGRE ALLE DETEKSJONER (både kjente og ukjente)
            if name == "Unknown":
                # Unknown: Lagre med cooldown
                if current_time - LAST_UNKNOWN_SAVE_TIME > UNKNOWN_SAVE_INTERVAL:
                    filename = save_detection(face_img, name)
                    log_event(f"🚨 UKJENT person detektert: {filename}")
                    LAST_UNKNOWN_SAVE_TIME = current_time
            else:
                # Kjente: Lagre med cooldown
                last_seen = last_detection_time.get(name, 0)
                if current_time - last_seen > DETECTION_COOLDOWN:
                    filename = save_detection(face_img, name, confidence)
                    log_event(f"✅ {name} detektert (conf: {confidence:.1f})")
                    last_detection_time[name] = current_time
        
        # Konverter til JPEG og stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- FLASK RUTER ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/database/<filename>")
def serve_database_image(filename):
    """Server bilder fra database-mappen"""
    return send_from_directory(DATASET_DIR, filename)

@app.route("/detections/<filename>")
def serve_detection_image(filename):
    """Server bilder fra detections-mappen"""
    return send_from_directory(DETECTIONS_DIR, filename)

@app.route("/list_unknowns", methods=["GET"])
def list_unknowns():
    """Returnerer liste over ukjente ansikter"""
    try:
        unknown_files = [f for f in os.listdir(DATASET_DIR) 
                         if f.startswith('Unknown_') and f.endswith('.jpg')]
        unknown_files.sort(reverse=True)  # Nyeste først
        log_event(f"📋 list_unknowns returnerer {len(unknown_files)} filer")
        return jsonify(unknown_files)
    except Exception as e:
        log_event(f"❌ Feil i list_unknowns: {e}")
        return jsonify([])

@app.route("/list_detections", methods=["GET"])
def list_detections():
    """Returnerer liste over alle deteksjoner"""
    detection_files = [f for f in os.listdir(DETECTIONS_DIR) if f.endswith('.jpg')]
    detection_files.sort(reverse=True)  # Nyeste først
    
    # Grupper etter person
    detections_by_person = {}
    for file in detection_files:
        person_name = file.rsplit('_', 1)[0]
        if person_name not in detections_by_person:
            detections_by_person[person_name] = []
        detections_by_person[person_name].append(file)
    
    return jsonify(detections_by_person)

@app.route("/register_unknown", methods=["POST"])
def register_unknown():
    """Registrerer og omdøper et ukjent bilde"""
    old_filename = request.form.get("filename")
    new_name = request.form.get("name")
    
    if not old_filename or not new_name:
        return jsonify({"error": "Filnavn og navn må oppgis"})

    if not old_filename.startswith("Unknown_"):
        return jsonify({"error": "Kan kun registrere Unknown-filer"})

    old_path = os.path.join(DATASET_DIR, old_filename)
    if not os.path.exists(old_path):
        return jsonify({"error": f"Filen finnes ikke"})

    # Lag nytt filnavn - behold timestamp
    timestamp = old_filename.replace("Unknown_", "")
    new_filename = f"{new_name}_{timestamp}"
    new_path = os.path.join(DATASET_DIR, new_filename)
    
    try:
        os.rename(old_path, new_path)
        log_event(f"✅ Registrert: {old_filename} → {new_name}")
        load_names_map()
        return jsonify({"success": True, "new_file": new_filename})
    except Exception as e:
        return jsonify({"error": f"Feil: {e}"})

@app.route("/train", methods=["POST"])
def train():
    """Trener modellen på alle registrerte ansikter"""
    global model_trained
    
    if not recognizer:
        return jsonify({"error": "Face recognizer ikke tilgjengelig"})
    
    load_names_map()
    
    faces = []
    labels = []
    
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.jpg')]
    
    for file in files:
        if file.startswith("Unknown_"):
            continue
            
        name = file.rsplit('_', 1)[0]
        
        person_id = None
        for pid, pname in id_to_name.items():
            if pname == name:
                person_id = pid
                break
        
        if person_id:
            img_path = os.path.join(DATASET_DIR, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(person_id)
    
    if len(faces) > 0:
        # VIKTIG: Anbefaler minst 5 bilder per person for god gjenkjenning
        images_per_person = {}
        for label in labels:
            images_per_person[label] = images_per_person.get(label, 0) + 1
        
        min_images = min(images_per_person.values())
        
        recognizer.train(faces, np.array(labels))
        recognizer.save(TRAINER_FILE)
        model_trained = True
        
        message = f"Modellen er trent! {len(faces)} bilder av {len(id_to_name)} personer."
        
        if min_images < 5:
            message += f" ⚠️ Tips: Noen personer har bare {min_images} bilder. Legg til flere bilder (5-10 per person) for bedre gjenkjenning!"
        
        log_event(f"✅ Modell trent med {len(faces)} bilder av {len(id_to_name)} personer")
        return jsonify({
            "success": True, 
            "images": len(faces),
            "persons": len(id_to_name),
            "min_images_per_person": min_images,
            "message": message
        })
    else:
        return jsonify({"error": "Ingen bilder funnet. Registrer personer først!"})

@app.route("/delete_unknown", methods=["POST"])
def delete_unknown():
    """Sletter et ukjent bilde"""
    filename = request.form.get("filename")
    
    if not filename or not filename.startswith("Unknown_"):
        return jsonify({"error": "Ugyldig filnavn"})
    
    filepath = os.path.join(DATASET_DIR, filename)
    
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            log_event(f"🗑️  Slettet: {filename}")
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Filen finnes ikke"})
    except Exception as e:
        return jsonify({"error": f"Feil: {e}"})

@app.route("/stats", methods=["GET"])
def get_stats():
    """Returnerer statistikk"""
    known_persons = len(id_to_name)
    unknown_count = len([f for f in os.listdir(DATASET_DIR) if f.startswith("Unknown_")])
    total_detections = len([f for f in os.listdir(DETECTIONS_DIR) if f.endswith(".jpg")]) if os.path.exists(DETECTIONS_DIR) else 0
    
    return jsonify({
        "known_persons": known_persons,
        "unknown_faces": unknown_count,
        "total_detections": total_detections,
        "model_trained": model_trained
    })

@app.route("/history", methods=["GET"])
def get_history():
    """Returnerer komplett historikk med alle deteksjoner sortert etter tid"""
    all_detections = []
    
    # Hent alle bilder fra detections-mappen
    if os.path.exists(DETECTIONS_DIR):
        for filename in os.listdir(DETECTIONS_DIR):
            if filename.endswith('.jpg'):
                # Parse filnavn: Navn_YYYYMMDD_HHMMSS.jpg
                parts = filename.replace('.jpg', '').split('_')
                if len(parts) >= 3:
                    name = '_'.join(parts[:-2])  # Håndterer navn med underscore
                    date_str = parts[-2]  # YYYYMMDD
                    time_str = parts[-1]  # HHMMSS
                    
                    # Konverter til lesbart format
                    try:
                        date_obj = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                        
                        all_detections.append({
                            'filename': filename,
                            'name': name,
                            'datetime': date_obj.strftime("%Y-%m-%d %H:%M:%S"),
                            'date': date_obj.strftime("%d.%m.%Y"),
                            'time': date_obj.strftime("%H:%M:%S"),
                            'timestamp': date_obj.timestamp()  # For sortering
                        })
                    except:
                        pass
    
    # Sorter etter tid (nyeste først)
    all_detections.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify(all_detections)
def get_stats():
    """Returnerer statistikk"""
    known_persons = len(id_to_name)
    unknown_count = len([f for f in os.listdir(DATASET_DIR) if f.startswith("Unknown_")])
    total_detections = len([f for f in os.listdir(DETECTIONS_DIR) if f.endswith(".jpg")])
    
    return jsonify({
        "known_persons": known_persons,
        "unknown_faces": unknown_count,
        "total_detections": total_detections,
        "model_trained": model_trained
    })

# --- START SERVEREN ---
if __name__ == "__main__":
    load_names_map()
    log_event("🚀 Face Recognition System starter...")
    log_event(f"📁 Database: {os.path.abspath(DATASET_DIR)}")
    log_event(f"📁 Detections: {os.path.abspath(DETECTIONS_DIR)}")
    
    if os.path.exists(TRAINER_FILE):
        log_event("✅ Treningsfil funnet")
    else:
        log_event("⚠️  Ingen treningsfil - modellen må trenes")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
