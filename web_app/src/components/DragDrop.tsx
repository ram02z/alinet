import React, { useEffect, useState } from 'react'
import { AiOutlineCloudUpload } from 'react-icons/ai'
import './DragDrop.css'

interface DragNdropProps {
  onFilesSelected: any
}

const DragNdrop: React.FC<DragNdropProps> = ({ onFilesSelected }) => {
  const [files, setFiles] = useState<File[]>([])

  const handleFileChange = (event: any) => {
    const selectedFiles = event.target.files
    if (selectedFiles && selectedFiles.length > 0) {
      const newFiles: File[] = Array.from(selectedFiles)
      setFiles((prevFiles: File[]) => [...prevFiles, ...newFiles])
    }
  }

  const handleDrop = (event: any) => {
    event.preventDefault()
    const droppedFiles = event.dataTransfer.files
    if (droppedFiles.length > 0) {
      const newFiles: File[] = Array.from(droppedFiles)
      setFiles((prevFiles: File[]) => [...prevFiles, ...newFiles])
    }
  }

  useEffect(() => {
    onFilesSelected(files)
    console.log(files)
  }, [files, onFilesSelected])

  return (
    <div
      className={`document-uploader`}
      onDrop={handleDrop}
      onDragOver={(event) => {
        event.preventDefault()
      }}
    >
      <>
        <div className="upload-info">
          <AiOutlineCloudUpload />
          <div>
            <p>
              Drag and drop your files here or
              <label
                htmlFor="browse-input"
                className="browse-btn"
              >
                Browse files
              </label>
            </p>
          </div>
        </div>

        <input
          type="file"
          hidden
          id="browse-input"
          onChange={handleFileChange}
          accept="video/*"
          multiple
        />
      </>
    </div>
  )
}

export default DragNdrop
