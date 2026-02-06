# Around line 120-130, REPLACE this section:
if state == STATE_SPOOF:
    
    # DEBUG: Save the face crop to see what's being fed to the model
    face_crop_yolo = frame[y1:y2, x1:x2]  # YOLO's crop
    face_crop_enhanced = crop_face_for_spoof(frame, (x1, y1, x2, y2))  # Our crop
    
    # Save for inspection
    if face_crop_yolo.size > 0:
        cv2.imwrite("debug_yolo_crop.jpg", face_crop_yolo)
    if face_crop_enhanced is not None:
        cv2.imwrite("debug_enhanced_crop.jpg", face_crop_enhanced)
    
    print(f"[DEBUG] YOLO bbox size: {x2-x1}x{y2-y1}")
    if face_crop_enhanced is not None:
        print(f"[DEBUG] Enhanced crop size: {face_crop_enhanced.shape}")
    else:
        print(f"[DEBUG] Enhanced crop: None (rejected)")
    
    # Your existing code:
    is_real, score = antispoof.predict_from_bbox(
        frame,
        (x1, y1, x2, y2),
        track_id=current_track_id
    )
    
    print(f"[DEBUG] Result: {'REAL' if is_real else 'SPOOF'}, Score: {score:.3f}")