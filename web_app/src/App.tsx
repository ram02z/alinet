import '@mantine/core/styles.css'
import { MantineProvider } from '@mantine/core'
import { theme } from './theme'
import { DragDrop } from './components/DragDrop'
import { FileList } from './components/FileList'
import { useState } from 'react'
import { Button } from '@mantine/core'
import { IconSettingsCog } from '@tabler/icons-react'

import './App.css'

export default function App() {
  const [files, setFiles] = useState<File[]>([])

  const generateQuestions = async () => {
    const formData = new FormData()
    files.forEach((file) => {
      console.log(file)
      formData.append('files', file)
    })

    console.log(files)

    // try {
    //   const response = await fetch('someurl', {
    //     method: 'POST',
    //     body: formData,
    //   })
    // } catch (error) {
    //   console.error(error)
    // }
  }

  return (
    <MantineProvider theme={theme}>
      <div className="body">
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

        <div className="generate-section">
          <Button onClick={() => generateQuestions}>
            <IconSettingsCog /> Generate Questions
          </Button>
        </div>
      </div>
    </MantineProvider>
  )
}
