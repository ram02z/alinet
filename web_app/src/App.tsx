import '@mantine/core/styles.css'
import { MantineProvider } from '@mantine/core'
import { theme } from './theme'
import { DragDrop } from './components/DragDrop'
import { FileList } from './components/FileList'
import { useState } from 'react'
import './App.css'

export default function App() {
  const [files, setFiles] = useState<File[]>([])

  return (
    <MantineProvider theme={theme}>
      <div className="upload-section">
        <DragDrop
          files={files}
          setFiles={setFiles}
        />

        <FileList
          files={files}
          setFiles={setFiles}
        />
      </div>
    </MantineProvider>
  )
}
